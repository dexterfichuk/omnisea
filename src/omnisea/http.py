"""Shared HTTP plumbing: one retrying session, OGC paging, time chunking, bounded threads.

Everything that talks to the network goes through here so that retry policy, the User-Agent and
the payload ceilings are set in exactly one place.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import PayloadTooLargeError, UpstreamError

__all__ = [
    "get_json",
    "set_max_concurrency",
    "get_session",
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


def get_session() -> requests.Session:
    """The process-wide session: connection pooling plus retries on 429 and 5xx."""
    global _session
    if _session is None:
        s = requests.Session()
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
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": _user_agent(), "Accept": "application/json"})
        _session = s
    return _session


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
