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
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import MissingDependencyError, PayloadTooLargeError, UpstreamError

__all__ = [
    "get_json",
    "set_max_concurrency",
    "get_session",
    "enable_cache",
    "disable_cache",
    "CACHE_POLICY",
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

    return f"omnisea/{__version__} (+https://github.com/omnisea/omnisea) python-requests"


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

#: How long a response stays usable, by URL. Rules are tried in order and the first match wins.
#:
#: The judgment encoded here is that **catalogues are near-static and measurements are not**.
#: IWLS ``GET /stations`` returns all 1573 stations in one ~2 MB payload and changes only when a
#: station is commissioned, moved or retired — months apart — so re-fetching it every process is
#: pure waste. A water level from one of those stations is minutes old, and serving a stale one
#: is a wrong number rather than a slow one, which is a correctness bug and not a performance
#: trade-off. So the volatile endpoints are listed first and excluded outright, and anything not
#: named here is left uncached by default.
#:
#: Patterns are matched with ``fnmatch`` against the URL minus its scheme, and the wildcards
#: cross ``/`` — hence the ordering, and hence the one anchored regex.
CACHE_POLICY: dict[str | re.Pattern[str], Any] = {
    # Measurements. These are the endpoints a stale hit would corrupt, so they are excluded
    # explicitly rather than by omission: the exclusion then survives a caller who passes
    # expire_after= to cache everything else.
    "api-iwls.dfo-mpo.gc.ca/api/v1/stations/*/data": NEVER_CACHE,
    "*/collections/swob-realtime/items": NEVER_CACHE,
    "*/collections/hydrometric-realtime/items": NEVER_CACHE,
    # Which stations exist and where they are. Anchored on the query string or end of URL so
    # that only the collection itself matches; the glob form would append '**' and quietly
    # swallow every future per-station endpoint, including the next volatile one.
    re.compile(r"api-iwls\.dfo-mpo\.gc\.ca/api/v1/stations(\?|$)"): timedelta(days=7),
    "api-iwls.dfo-mpo.gc.ca/api/v1/stations/*/metadata": timedelta(days=7),
    "*/collections/*-stations/items": timedelta(days=7),
    # Published metadata records, which CIOOS reads a few hundred files at a time out of a
    # GitHub repository. Unauthenticated GitHub allows 60 requests an hour, so this is the
    # difference between discovery working twice in a row and not.
    "api.github.com/repos/": timedelta(days=1),
    "raw.githubusercontent.com/": timedelta(days=1),
    # Continuously-appended archives. These are quality-controlled and published days behind
    # real time, so an hour of staleness cannot hide a measurement that was available when the
    # query ran — short enough to stay honest, long enough that re-running a notebook cell is
    # free.
    "*/collections/climate-hourly/items": timedelta(hours=1),
    "*/collections/climate-daily/items": timedelta(hours=1),
    "*/collections/climate-monthly/items": timedelta(hours=1),
    "*/collections/hydrometric-daily-mean/items": timedelta(hours=1),
    "*/collections/hydrometric-monthly-mean/items": timedelta(hours=1),
    # Annual products. AHCCD is re-issued about once a year and the hydrometric annual
    # summaries are computed after the water year closes, so an hour would be pointlessly
    # short. ``ahccd-stations`` is not caught here — the station rule above already claimed it.
    "*/collections/ahccd-*/items": timedelta(days=1),
    "*/collections/hydrometric-annual-*/items": timedelta(days=1),
}


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
    """Caller rules first (first match wins, so they can override), then :data:`CACHE_POLICY`."""
    table: dict[Any, Any] = {k: _expiry(requests_cache, v) for k, v in (overrides or {}).items()}
    for pattern, value in CACHE_POLICY.items():
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

    What is cached, and for how long, is decided per URL by :data:`CACHE_POLICY` — station
    catalogues and station metadata for a week, published metadata records for a day, the ECCC
    climate archive for an hour, and realtime observations never. Requires the ``cache`` extra
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
        urls_expire_after: extra ``{pattern: expiry}`` rules, matched *before* the built-in
            ones so they can override them.
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


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    provider: str | None = None,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
) -> Any:
    """GET and decode JSON, converting any failure into an :class:`UpstreamError`."""
    session = get_session()
    log.debug("GET %s params=%s", url, params)
    slots = _request_slots
    with slots:
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            raise UpstreamError(
                f"request to {url} failed: {exc}", provider=provider, url=url
            ) from exc

    if not resp.ok:
        raise UpstreamError(
            "upstream request failed",
            provider=provider,
            url=resp.url,
            status=resp.status_code,
            detail=_extract_detail(resp),
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise UpstreamError(
            f"upstream returned non-JSON body: {(resp.text or '')[:200]!r}",
            provider=provider,
            url=resp.url,
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
        if returned == 0 or returned < page_size:
            return
        offset += returned
        if isinstance(matched, int) and offset >= matched:
            return


def chunk_time(
    start: pd.Timestamp, end: pd.Timestamp, *, max_days: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, end]`` into windows no longer than ``max_days``.

    Chunks share their boundary instants (chunk N ends where chunk N+1 begins) because these APIs
    treat both endpoints as inclusive; the duplicate rows are removed when the frames are
    concatenated, which is cheaper than trying to nudge the boundaries by an epsilon.
    """
    if end <= start:
        return []
    span = pd.Timedelta(days=max_days)
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
