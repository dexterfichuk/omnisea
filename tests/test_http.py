"""The HTTP layer's concurrency discipline — the part 825 other tests never touched.

The two-level semaphore is the fix for a production incident (CIOOS Pacific answering 413
when eight requests landed at once), and until this file nothing would fail if a rewrite
quietly dropped the per-host cap. These tests hold real threads at a barrier inside a stub
transport and count what got through.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from omnisea import http
from omnisea.errors import QueryError, UpstreamError


class Counter:
    """Track the high-water mark of concurrent calls, per key."""

    def __init__(self):
        self.lock = threading.Lock()
        self.live: dict[str, int] = {}
        self.peak: dict[str, int] = {}

    def enter(self, key):
        with self.lock:
            self.live[key] = self.live.get(key, 0) + 1
            self.peak[key] = max(self.peak.get(key, 0), self.live[key])

    def leave(self, key):
        with self.lock:
            self.live[key] -= 1


class FakeResponse:
    ok = True
    status_code = 200
    text = "body"
    content = b"body"
    url = "http://x"
    headers: dict = {"Content-Length": "4"}

    def json(self):
        return {"ok": True}

    def iter_content(self, *a, **k):
        return iter([b"body"])


def slow_session(counter, hold=0.05):
    import time

    class Session:
        def get(self, url, params=None, timeout=None):
            host = url.split("//", 1)[-1].split("/", 1)[0]
            counter.enter(host)
            counter.enter("*")
            time.sleep(hold)
            counter.leave(host)
            counter.leave("*")
            return FakeResponse()

    return Session()


class TestTwoLevelConcurrency:
    def test_no_more_than_the_per_host_cap_reaches_one_host(self):
        counter = Counter()
        with patch.object(http, "get_session", lambda: slow_session(counter)):
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(
                    lambda i: http.get_text("https://one.example/a", None),
                    range(16),
                ))
        assert counter.peak["one.example"] <= http.DEFAULT_MAX_PER_HOST, (
            "the per-host courtesy cap is the fix for a real 413 incident"
        )

    def test_different_hosts_run_wider_than_one_host_may(self):
        counter = Counter()
        hosts = [f"https://h{i}.example/x" for i in range(12)]
        with patch.object(http, "get_session", lambda: slow_session(counter)):
            with ThreadPoolExecutor(max_workers=24) as pool:
                list(pool.map(lambda u: http.get_text(u, None), hosts * 2))
        assert counter.peak["*"] > http.DEFAULT_MAX_PER_HOST, (
            "the global pool must be wider than one host's share, or fifteen "
            "institutions are rationed as if they were one"
        )
        assert counter.peak["*"] <= http.DEFAULT_MAX_CONCURRENT_REQUESTS

    def test_set_max_concurrency_rejects_nonsense(self):
        with pytest.raises(ValueError):
            http.set_max_concurrency(0)
        with pytest.raises(ValueError):
            http.set_max_concurrency(4, per_host=0)

    def test_set_max_concurrency_changes_both_levels(self):
        try:
            http.set_max_concurrency(2, per_host=1)
            counter = Counter()
            with patch.object(http, "get_session", lambda: slow_session(counter)):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(
                        lambda i: http.get_text("https://solo.example/a", None),
                        range(8),
                    ))
            assert counter.peak["solo.example"] == 1
        finally:
            http.set_max_concurrency(
                http.DEFAULT_MAX_CONCURRENT_REQUESTS, per_host=http.DEFAULT_MAX_PER_HOST
            )


class TestMapThreads:
    def test_results_come_back_in_input_order(self):
        import time

        def work(n):
            time.sleep(0.02 if n == 0 else 0)  # the first input finishes last
            return n * 10

        assert http.map_threads(work, [0, 1, 2, 3], max_workers=4) == [0, 10, 20, 30]

    def test_a_worker_exception_propagates(self):
        def work(n):
            if n == 2:
                raise UpstreamError("boom", provider="t")
            return n

        with pytest.raises(UpstreamError):
            http.map_threads(work, [0, 1, 2], max_workers=3)


class TestGetText:
    def test_the_real_body_runs_and_returns_text(self):
        counter = Counter()
        with patch.object(http, "get_session", lambda: slow_session(counter, hold=0)):
            assert http.get_text("https://x.example/t", {"a": "1"}) == "body"

    def test_an_http_error_is_an_upstream_error(self):
        class Bad(FakeResponse):
            ok = False
            status_code = 503

        class Session:
            def get(self, url, params=None, timeout=None):
                return Bad()

        with patch.object(http, "get_session", lambda: Session()):
            with pytest.raises(UpstreamError):
                http.get_text("https://x.example/t", None)


class TestKeywordLiteralsRaiseQueryError:
    """The four validation sites the taxonomy pass converted: still ValueError-compatible."""

    def test_on_error_and_nearest_still_catchable_as_value_error(self):
        import omnisea

        cat = omnisea.Catalog(matches=[], query=None, errors={}, notes={})
        with pytest.raises(QueryError):
            cat.fetch(on_error="explode")
        with pytest.raises(ValueError):  # QueryError subclasses ValueError; old catches hold
            cat.filter(nearest=0)
