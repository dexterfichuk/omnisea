"""Shared HTTP plumbing: one retrying session, OGC paging, time chunking, bounded threads.

Everything that talks to the network goes through here so that retry policy, the User-Agent and
the payload ceilings are set in exactly one place. The optional response cache
(:func:`enable_cache`) is a layer in front of that same session rather than a second way of
making requests, so nothing downstream has to know whether an answer came off disk.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import MissingDependencyError, PayloadTooLargeError, UpstreamError

__all__ = [
    "get_json",
    "redact_url",
    "redact_params",
    "SENSITIVE_PARAMS",
    "set_max_concurrency",
    "get_session",
    "enable_cache",
    "disable_cache",
    "CACHE_POLICY",
    "cache_policy",
    "NEVER_CACHE",
    "paginate_ogc_items",
    "chunk_time",
    "map_threads",
    "DEFAULT_MAX_WORKERS",
]

log = logging.getLogger("omnisea.http")

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_WORKERS = 8
DEFAULT_TIMEOUT = (10, 120)  # (connect, read) seconds
OGC_PAGE_SIZE = 10_000  # pygeoapi accepts large limits; fewer round-trips than the 10-row default

_session: requests.Session | None = None

#: Ceiling on *simultaneous HTTP requests*, enforced here rather than at each thread pool.
#: Discovery fans out across providers and each provider fans out across stations, so bounding
#: the pools individually would still allow workers x providers requests in flight at once and
#: get us rate-limited. Bounding the scarce resource itself makes the limit hold however deeply
#: the call sites nest.
DEFAULT_MAX_CONCURRENT_REQUESTS = 8
_request_slots = threading.BoundedSemaphore(DEFAULT_MAX_CONCURRENT_REQUESTS)


def set_max_concurrency(n: int) -> None:
    """Change the global cap on simultaneous HTTP requests."""
    global _request_slots
    if n < 1:
        raise ValueError(f"max concurrency must be >= 1; got {n}")
    _request_slots = threading.BoundedSemaphore(n)


def _user_agent() -> str:
    from . import __version__

    return f"omnisea/{__version__} (+https://github.com/dexterfichuk/omnisea) python-requests"


def _configure(session: requests.Session) -> requests.Session:
    """Attach the retry policy and headers that every omnisea request depends on.

    Split out from :func:`get_session` so that a cached session is set up by exactly the same
    code. Retries live on the mounted adapter, below the cache, which is where they belong: a
    cache hit does not need retrying and a miss still gets the full backoff ladder.
    """
    retry = Retry(
        total=4,
        connect=3,
        read=3,
        status=4,
        backoff_factor=0.7,  # 0.7s, 1.4s, 2.8s, 5.6s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": _user_agent(), "Accept": "application/json"})
    return session


def get_session() -> requests.Session:
    """The process-wide session: connection pooling plus retries on 429 and 5xx.

    Returns a ``requests_cache.CachedSession`` instead once :func:`enable_cache` has been called.
    """
    global _session
    if _session is None:
        _session = _configure(requests.Session())
    return _session


# --------------------------------------------------------------------------- response cache

#: Marker for "do not store this response at all", translated to requests-cache's own sentinel
#: when the cache is built. Spelled locally so :data:`CACHE_POLICY` can be read — and written to
#: by callers — without importing an optional dependency.
NEVER_CACHE = "omnisea:never-cache"

#: Core fallback rules, tried after every provider's own. Rules are ordered ``{pattern:
#: expiry}`` pairs, first match wins; patterns are matched with ``fnmatch`` against the URL
#: minus its scheme, and the wildcards cross ``/``.
#:
#: The endpoint-specific judgment — which URLs serve near-static catalogues and which serve
#: measurements a stale hit would corrupt — lives on each :class:`~omnisea.Provider` as its
#: ``cache_policy``, because the provider is the party that knows. Built-in and third-party
#: providers declare it identically, and :func:`enable_cache` merges every registered
#: provider's rules. This table is deliberately empty: an endpoint nobody has claimed is
#: assumed to serve measurements, because guessing wrong in that direction is only slow.
CACHE_POLICY: dict[str | re.Pattern[str], Any] = {}


def cache_policy() -> dict[str | re.Pattern[str], Any]:
    """The merged cache policy: every registered provider's rules, then the core fallbacks.

    Providers merge in registry order with each provider's internal ordering preserved, which
    is what matters — a provider lists its volatile endpoints before its broad catalogue
    globs, and its patterns name its own hosts, so rules from different providers do not
    contend for the same URL.
    """
    # Imported here rather than at module top: the registry imports the provider base class,
    # which imports this module for get_json. By the time anyone can call enable_cache the
    # package is fully imported, so the late import is safe and breaks the cycle.
    from .registry import all_providers

    table: dict[str | re.Pattern[str], Any] = {}
    for provider in all_providers():
        for pattern, value in (provider.cache_policy or {}).items():
            table.setdefault(pattern, value)
    for pattern, value in CACHE_POLICY.items():
        table.setdefault(pattern, value)
    return table


def _import_requests_cache() -> Any:
    """Import requests-cache, or say which extra provides it.

    The bare ModuleNotFoundError names a package the user never asked for by name; the extra is
    what they actually need to type.
    """
    try:
        import requests_cache
    except ImportError as exc:
        raise MissingDependencyError(
            "requests-cache", "cache", "for response caching"
        ) from exc
    return requests_cache


def _expiry(requests_cache: Any, value: Any) -> Any:
    """Translate :data:`NEVER_CACHE` into requests-cache's own sentinel; pass anything else on."""
    return requests_cache.DO_NOT_CACHE if value == NEVER_CACHE else value


def _expiry_table(
    requests_cache: Any, overrides: Mapping[str | re.Pattern[str], Any] | None
) -> dict[Any, Any]:
    """Caller rules first (first match wins, so they can override), then :func:`cache_policy`."""
    table: dict[Any, Any] = {k: _expiry(requests_cache, v) for k, v in (overrides or {}).items()}
    for pattern, value in cache_policy().items():
        table.setdefault(pattern, _expiry(requests_cache, value))
    return table


def enable_cache(
    path: str | Path | None = None,
    *,
    expire_after: Any = NEVER_CACHE,
    urls_expire_after: Mapping[str | re.Pattern[str], Any] | None = None,
    backend: str = "sqlite",
) -> requests.Session:
    """Route omnisea's requests through an on-disk response cache. Off unless you call this.

    What is cached, and for how long, is decided per URL by each registered provider's
    ``cache_policy`` (see :func:`cache_policy` for the merged table) — station catalogues and
    station metadata for a week, published metadata records for a day, the ECCC climate archive
    for an hour, and realtime observations never. Requires the ``cache`` extra
    (``pip install "omnisea[cache]"``).

    Call it once at startup: it replaces the process-wide session, so calling it while a fetch
    is in flight leaves those requests on the old one. The new session keeps the same retry
    ladder and User-Agent, and cache hits still take a concurrency slot in :func:`get_json`, so
    the request ceiling holds however an answer is served.

    Providers that memoize a catalogue in-process — DFO keeps the IWLS station list in a module
    global — go on doing so, and short-circuit before reaching here. This cache sits underneath
    that and is what makes the *first* call in each new process cheap; the two layers stack
    rather than duplicate, which is also why the in-process one still works without this extra.

    Args:
        path: where to keep the sqlite database. ``None`` uses the platform cache directory —
            ``~/Library/Caches/omnisea/http.sqlite`` on macOS, ``~/.cache/omnisea/http.sqlite``
            on Linux.
        expire_after: fallback expiry for URLs no rule matches. Defaults to
            :data:`NEVER_CACHE` — an unrecognised endpoint is assumed to serve measurements,
            because guessing wrong in that direction is only slow. Pass seconds or a
            ``timedelta`` to cache the rest too.
        urls_expire_after: extra ``{pattern: expiry}`` rules, matched *before* the providers'
            own so they can override them.
        backend: any requests-cache backend; ``"memory"`` caches for the life of the process
            and writes nothing to disk.

    Returns the session now in use.
    """
    requests_cache = _import_requests_cache()

    backend_kwargs: dict[str, Any] = {}
    if path is not None:
        cache_name = str(path)
    else:
        cache_name = "omnisea/http"
        if backend == "sqlite":
            # Otherwise requests-cache drops the database in the working directory, which for a
            # library invoked from a notebook is somebody's project folder.
            backend_kwargs["use_cache_dir"] = True

    session = requests_cache.CachedSession(
        cache_name=cache_name,
        backend=backend,
        expire_after=_expiry(requests_cache, expire_after),
        urls_expire_after=_expiry_table(requests_cache, urls_expire_after),
        # Deliberately ignore the upstreams' own Cache-Control. IWLS answers *every* request
        # with "no-cache, no-store, max-age=0, must-revalidate", including the station list
        # that changes a few times a year, so honouring it would mean caching nothing at all.
        # CACHE_POLICY overrides that for catalogue endpoints only, and the endpoints where
        # freshness genuinely matters are excluded there anyway.
        cache_control=False,
        # Never replay a failure: a 500 stored for a week would look like a dead provider long
        # after it recovered, and a 404 would hide a station that has since appeared.
        allowable_codes=(200,),
        **backend_kwargs,
    )

    global _session
    previous = _session
    _session = _configure(session)
    if previous is not None:
        previous.close()
    log.debug("response cache enabled: %s (%s)", cache_name, backend)
    return _session


def disable_cache(*, clear: bool = False) -> None:
    """Go back to an uncached session; the next request builds one. A no-op if none is enabled.

    ``clear=True`` also empties the stored responses, which is the way to force a re-fetch of
    something :data:`CACHE_POLICY` still considers fresh.
    """
    global _session
    session = _session
    # Duck-typed rather than an isinstance check so that disabling a cache that was never
    # enabled does not import the optional dependency just to answer "no".
    if session is None or not hasattr(session, "cache"):
        return
    if clear:
        session.cache.clear()
    session.close()
    _session = None
    log.debug("response cache disabled")


def _scrub_secrets(text: str | None, params: Mapping[str, Any] | None) -> str | None:
    """Remove the secret values *we sent* from text an upstream sent back.

    Redacting the URL is not enough on its own. Ocean Networks Canada answers a bad request by
    quoting a corrected URL — with the caller's token still in it — inside the JSON error body,
    which then lands in an exception message, a log, and any traceback a user pastes into an
    issue. Since this function knows exactly which values were secret, it removes those literal
    strings wherever they appear rather than trying to guess a format.
    """
    if not text or not params:
        return text
    for key, value in params.items():
        if str(key).lower() not in SENSITIVE_PARAMS:
            continue
        secret = str(value)
        if len(secret) >= 6 and secret in text:
            text = text.replace(secret, _REDACTED)
    return text


def _extract_detail(resp: requests.Response) -> str | None:
    """Pull the human-readable complaint out of an error body.

    IWLS and pygeoapi both return JSON errors but disagree on the field names, so try the ones
    they actually use before falling back to raw text.
    """
    try:
        body = resp.json()
    except ValueError:
        text = (resp.text or "").strip()
        return text[:400] or None
    if isinstance(body, dict):
        for key in ("errors", "error", "message", "description", "detail", "title"):
            if key in body and body[key]:
                val = body[key]
                if isinstance(val, list):
                    return "; ".join(str(v) for v in val)[:400]
                return str(val)[:400]
    return str(body)[:400]


#: Query parameters that carry a secret. Their values are replaced with ``REDACTED`` wherever
#: omnisea prints, logs or stores a URL.
#:
#: Some services authenticate in the query string rather than a header — Ocean Networks Canada
#: takes ``?token=``. A URL like that reaches three places that outlive the request: the debug
#: log, the message of every :class:`~omnisea.errors.UpstreamError`, and the ``source_url``
#: attribute stamped on each node, which is then written into netCDF files people share and
#: commit. Leaking a credential that way is the kind of mistake nobody notices until it is
#: published, so redaction happens here, once, for every provider.
SENSITIVE_PARAMS = frozenset({"token", "apptoken", "api_key", "apikey", "auth", "key",
                              "access_token", "password", "secret"})

_REDACTED = "REDACTED"


def redact_url(url: str) -> str:
    """A URL safe to log, raise or store: secret query values replaced with ``REDACTED``."""
    try:
        parts = urlsplit(str(url))
    except ValueError:  # pragma: no cover - defensive; urlsplit is very forgiving
        return str(url)
    if not parts.query:
        return str(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    if not any(k.lower() in SENSITIVE_PARAMS for k, _ in pairs):
        return str(url)
    cleaned = [(k, _REDACTED if k.lower() in SENSITIVE_PARAMS else v) for k, v in pairs]
    return urlunsplit(parts._replace(query=urlencode(cleaned)))


def redact_params(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The same, for a parameter mapping on its way to a log line."""
    if not params:
        return None if params is None else {}
    return {
        k: (_REDACTED if str(k).lower() in SENSITIVE_PARAMS else v) for k, v in params.items()
    }


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    provider: str | None = None,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
) -> Any:
    """GET and decode JSON, converting any failure into an :class:`UpstreamError`.

    Credentials passed in the query string are redacted from the log line and from every error
    this raises — see :data:`SENSITIVE_PARAMS`.
    """
    session = get_session()
    log.debug("GET %s params=%s", redact_url(url), redact_params(params))
    slots = _request_slots
    with slots:
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            safe = redact_url(url)
            raise UpstreamError(
                f"request to {safe} failed: {exc}", provider=provider, url=safe
            ) from exc

    if not resp.ok:
        raise UpstreamError(
            "upstream request failed",
            provider=provider,
            url=redact_url(resp.url),
            status=resp.status_code,
            detail=_scrub_secrets(_extract_detail(resp), params),
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise UpstreamError(
            f"upstream returned non-JSON body: {(resp.text or '')[:200]!r}",
            provider=provider,
            url=redact_url(resp.url),
            status=resp.status_code,
        ) from exc


def paginate_ogc_items(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    provider: str | None = None,
    max_items: int = 250_000,
    page_size: int = OGC_PAGE_SIZE,
) -> Iterator[dict[str, Any]]:
    """Yield GeoJSON features from an OGC API - Features collection, page by page.

    pygeoapi caps ``limit`` server-side, so paging uses explicit ``offset`` steps rather than
    trusting the ``next`` link. Exceeding ``max_items`` raises :class:`PayloadTooLargeError`
    instead of quietly stopping — a truncated series that *looks* complete is worse than an error.
    """
    base = dict(params or {})
    base.setdefault("f", "json")
    offset = 0
    seen = 0
    while True:
        page_params = dict(base, limit=page_size, offset=offset)
        payload = get_json(url, page_params, provider=provider)
        features = payload.get("features") or []
        matched = payload.get("numberMatched")

        if offset == 0 and isinstance(matched, int) and matched > max_items:
            raise PayloadTooLargeError(
                f"{url} matches {matched:,} features, over the {max_items:,} ceiling. "
                "Narrow the bbox/time window, or raise max_items.",
                estimate=matched,
                limit=max_items,
            )

        for feat in features:
            seen += 1
            if seen > max_items:
                raise PayloadTooLargeError(
                    f"{url} returned more than the {max_items:,} feature ceiling. "
                    "Narrow the bbox/time window, or raise max_items.",
                    estimate=seen,
                    limit=max_items,
                )
            yield feat

        returned = len(features)
        if returned == 0:
            return
        offset += returned
        if isinstance(matched, int) and offset >= matched:
            return
        # Deliberately NOT stopping on a short page. OGC API - Features permits a server to
        # return fewer items than `limit`, and pygeoapi caps `limit` server-side — which is the
        # premise of paging by explicit offset in the first place. Treating a short page as the
        # end made the *first* page the last one against a capped server: 1000 of 2500 stations
        # discovered, the numberMatched proving otherwise thrown away, and nothing anywhere
        # saying so. The cost of continuing is one extra request that comes back empty; the
        # cost of stopping was a silently truncated result that looked complete.


def chunk_time(
    start: pd.Timestamp, end: pd.Timestamp, *, max_days: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, end]`` into windows no longer than ``max_days``.

    Chunks share their boundary instants (chunk N ends where chunk N+1 begins) because these APIs
    treat both endpoints as inclusive; the duplicate rows are removed when the frames are
    concatenated, which is cheaper than trying to nudge the boundaries by an epsilon.
    """
    if max_days <= 0:
        raise ValueError(f"max_days must be positive; got {max_days!r}")
    if end <= start:
        return []
    span = pd.Timedelta(days=max_days)
    if span <= pd.Timedelta(0):
        # A positive max_days can still round to zero nanoseconds. Left alone the cursor never
        # advances and the loop below allocates chunks until the process dies.
        raise ValueError(f"max_days={max_days!r} is too small to form a time chunk")
    if end - start <= span:
        return [(start, end)]
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + span, end)
        out.append((cursor, stop))
        if stop >= end:
            break
        cursor = stop
    return out


def map_threads(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    label: str = "task",
) -> list[R]:
    """Run ``fn`` over ``items`` on a bounded pool, preserving input order.

    Results come back in submission order so that station ordering stays deterministic across
    runs. Exceptions propagate — a partial tree that silently dropped a station would be a
    scientific-correctness hazard.
    """
    items = list(items)
    if not items:
        return []
    if len(items) == 1 or max_workers <= 1:
        return [fn(item) for item in items]
    workers = min(max_workers, len(items))
    log.debug("running %d %s(s) across %d workers", len(items), label, workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="omnisea") as pool:
        return list(pool.map(fn, items))
