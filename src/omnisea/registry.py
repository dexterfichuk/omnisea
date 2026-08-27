"""Registry of providers and their data sources.

Built-in providers register themselves at import. Third-party packages advertise their own by
declaring an entry point::

    [project.entry-points."omnisea.providers"]
    my_org = "my_package.provider:MyOrgProvider"

The entry point may point at a :class:`~omnisea.providers.base.Provider` (registering all of its
sources) or at a single :class:`~omnisea.providers.base.DataSource`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .errors import UnknownProviderError
from .providers.base import DataSource, Provider

__all__ = [
    "register_provider",
    "register_source",
    "get_source",
    "get_provider",
    "source_names",
    "provider_names",
    "all_sources",
    "all_providers",
    "select",
    "known_variable_names",
]

log = logging.getLogger("omnisea.registry")


def _ensure_loaded() -> None:
    _load_entry_points()

_SOURCES: dict[str, DataSource] = {}
_PROVIDERS: dict[str, Provider] = {}
_ENTRY_POINTS_LOADED = False
ENTRY_POINT_GROUP = "omnisea.providers"


def register_source(source: DataSource, *, replace: bool = False) -> DataSource:
    """Add one data source to the registry, keyed by ``source.name``.

    Entry points are loaded first. Registering a source back-registers its provider, and doing
    that *before* the plugin advertising the same provider had loaded pre-empted it: the
    provider name was taken, its own sources never registered, and ``select("that_provider")``
    then raised "unknown provider 'x'; registered providers: ..., x" — naming it in the same
    breath as calling it unknown. Order-dependent, so it reproduced in a script and vanished in
    a notebook, which is worse than a consistent failure.
    """
    _ensure_loaded()
    if not source.name:
        raise ValueError(f"{source!r} must set a non-empty .name before registration")
    if source.name in _SOURCES and not replace:
        raise ValueError(
            f"source {source.name!r} is already registered; pass replace=True to override"
        )
    if source.name in _PROVIDERS and _PROVIDERS[source.name] is not source.provider:
        # select() resolves a source name before a provider name, so a source sharing an
        # organization's name would answer for the whole organization: asking for "acme" would
        # quietly return one of its datasets instead of all of them.
        raise ValueError(
            f"source {source.name!r} collides with the registered provider of the same name. "
            "Selecting it would be ambiguous — give the source a distinct name."
        )
    _SOURCES[source.name] = source
    provider = source.provider
    if provider is not None and provider.name:
        _PROVIDERS.setdefault(provider.name, provider)
    return source


def register_provider(provider: Provider, *, replace: bool = False) -> Provider:
    """Register an organization and every dataset it publishes."""
    _ensure_loaded()
    if not provider.name:
        raise ValueError(f"{provider!r} must set a non-empty .name before registration")
    if provider.name in _PROVIDERS and not replace:
        raise ValueError(
            f"provider {provider.name!r} is already registered; pass replace=True to override"
        )
    _PROVIDERS[provider.name] = provider
    for source in provider.sources:
        register_source(source, replace=replace)
    return provider


def _load_entry_points() -> None:
    """Import third-party providers once, tolerating individually broken plugins."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points

        found = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - a broken environment must not break the library
        log.debug("entry-point discovery unavailable", exc_info=True)
        return
    for ep in found:
        try:
            obj = ep.load()
            instance = obj() if isinstance(obj, type) else obj
            if isinstance(instance, Provider):
                if instance.name not in _PROVIDERS:
                    register_provider(instance)
            elif isinstance(instance, DataSource):
                if instance.name not in _SOURCES:
                    register_source(instance)
            else:
                log.warning(
                    "entry point %r is neither a Provider nor a DataSource: %r", ep.name, instance
                )
        except Exception:  # noqa: BLE001 - one bad plugin should not hide the rest
            log.warning("failed to load provider entry point %r", ep.name, exc_info=True)


def get_source(name: str) -> DataSource:
    _ensure_loaded()
    try:
        return _SOURCES[name]
    except KeyError:
        raise UnknownProviderError(name, sorted(_SOURCES) + sorted(_PROVIDERS)) from None


def get_provider(name: str) -> Provider:
    _ensure_loaded()
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise UnknownProviderError(name, sorted(_PROVIDERS)) from None


def source_names() -> list[str]:
    _ensure_loaded()
    return sorted(_SOURCES)


def provider_names() -> list[str]:
    _ensure_loaded()
    return sorted(_PROVIDERS)


def all_sources() -> list[DataSource]:
    _ensure_loaded()
    return [_SOURCES[n] for n in sorted(_SOURCES)]


def all_providers() -> list[Provider]:
    _ensure_loaded()
    return [_PROVIDERS[n] for n in sorted(_PROVIDERS)]


def select(names: Iterable[str] | None) -> list[DataSource]:
    """Resolve names to data sources.

    A name may be either a source (``"eccc_climate"``) or a whole organization (``"eccc"``), in
    which case every source it publishes is selected. Passing nothing selects everything.
    """
    _ensure_loaded()
    if not names:
        return all_sources()

    chosen: list[DataSource] = []
    seen: set[str] = set()
    for name in names:
        if name in _SOURCES:
            picked = [_SOURCES[name]]
        elif name in _PROVIDERS:
            # Only the provider's *registered* sources. An entry point may point at a single
            # DataSource, which back-registers its provider; Provider.sources would then build
            # every dataset that provider declares, and fetch() would later die on a source the
            # user never asked for and the registry never accepted.
            picked = [s for s in _PROVIDERS[name].sources if _SOURCES.get(s.name) is s]
            if not picked:
                raise UnknownProviderError(name, sorted(_SOURCES) + sorted(_PROVIDERS))
        else:
            raise UnknownProviderError(name, sorted(_SOURCES) + sorted(_PROVIDERS))
        for source in picked:
            if source.name not in seen:
                seen.add(source.name)
                chosen.append(source)
    return chosen


def known_variable_names() -> frozenset[str]:
    """Every name any registered source curates: CF standard names, omnisea variable names and
    raw provider field names.

    A name outside this set is not necessarily unavailable — it may be a field a platform
    publishes that omnisea carries through without a CF mapping. It only means omnisea cannot
    reason about which source holds it.
    """
    names: set[str] = set()
    for source in all_sources():
        for raw, spec in source.fields.items():
            names.add(raw)
            names.add(spec.var)
            if spec.standard_name:
                names.add(spec.standard_name)
    return frozenset(names)
