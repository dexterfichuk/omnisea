"""Exception hierarchy for omnisea.

Every failure a user can hit is an :class:`OmniseaError`, so ``except omnisea.OmniseaError``
is a complete catch. Upstream HTTP failures keep the provider name, URL and server-supplied
detail attached rather than surfacing a bare ``requests`` traceback.
"""

from __future__ import annotations

__all__ = [
    "OmniseaError",
    "QueryError",
    "ProviderError",
    "UnknownProviderError",
    "UpstreamError",
    "PayloadTooLargeError",
    "MissingDependencyError",
]


class OmniseaError(Exception):
    """Base class for every omnisea-raised error."""


class QueryError(OmniseaError, ValueError):
    """The query itself is malformed or self-contradictory."""


class ProviderError(OmniseaError):
    """A provider failed while discovering or fetching."""

    def __init__(self, message: str, *, provider: str | None = None):
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)


class UnknownProviderError(ProviderError, KeyError):
    """A provider name was requested that is not registered."""

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        # KeyError.__str__ would re-quote the message, so build it explicitly.
        ProviderError.__init__(
            self,
            f"unknown provider {name!r}; registered providers: {', '.join(available) or '(none)'}",
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.args[0]


class UpstreamError(ProviderError):
    """An upstream service returned an error, or was unreachable.

    The upstream's own message is preserved in :attr:`detail` — for example IWLS replies to an
    over-long window with ``date interval should not be bigger than 7 days``, which is far more
    useful than the bare status code.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        url: str | None = None,
        status: int | None = None,
        detail: str | None = None,
    ):
        self.url = url
        self.status = status
        self.detail = detail
        parts = [message]
        if status is not None:
            parts.append(f"(HTTP {status})")
        if detail:
            parts.append(f"- upstream said: {detail}")
        if url:
            parts.append(f"\n  url: {url}")
        super().__init__(" ".join(parts), provider=provider)


class MissingDependencyError(OmniseaError, ImportError):
    """An optional feature was used without the extra that provides it.

    Also an ``ImportError``, so code probing for optional dependencies with
    ``except ImportError`` keeps working, while ``except omnisea.OmniseaError`` stays a
    complete catch.
    """

    def __init__(self, package: str, extra: str, purpose: str = ""):
        self.package = package
        self.extra = extra
        detail = f" {purpose}" if purpose else ""
        super().__init__(
            f"{package} is required{detail}, but is not installed. "
            f'Install it with: pip install "omnisea[{extra}]"'
        )


class PayloadTooLargeError(OmniseaError):
    """A query would pull down more rows than the configured ceiling allows.

    Raised *before* the request goes out (from the row estimate) or while paging, so a wide
    query fails loudly instead of silently truncating or hammering the upstream API.
    """

    def __init__(self, message: str, *, estimate: int | None = None, limit: int | None = None):
        self.estimate = estimate
        self.limit = limit
        super().__init__(message)
