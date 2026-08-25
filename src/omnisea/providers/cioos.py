"""CIOOS — metadata records authored in the CIOOS metadata-entry-form.

``https://github.com/cioos-siooc/metadata-entry-form`` (records: ``https://form.cioos.ca``)

This is a **discovery** source, not a retrieval one. A CIOOS record is ISO 19115 metadata: it
says a dataset exists, where and when it was collected, which Essential Ocean Variables it
covers, and where to download it. It hands you a URL, not an array — so it contributes
:class:`~omnisea.Catalog` rows and nothing to the tree, exactly as the research doc concluded
for STAC.

**There is no single public endpoint.** The entry form publishes each organization's records to
that organization's own GitHub repository, and its Firebase database is not world-readable
(verified: the REST endpoint returns ``401 Permission denied``). So records are read from
wherever *you* keep them, via the ``cioos_records`` option:

.. code-block:: python

    omnisea.discover(bbox=..., time=..., cioos_records="./records")           # directory
    omnisea.discover(bbox=..., time=..., cioos_records="cioos-siooc/records") # GitHub repo
    omnisea.discover(bbox=..., time=..., cioos_records="https://.../records.json")

Both record layouts are understood: the raw form/Firebase shape (``title``, ``map``, ``eov``)
and the ``metadata-xml`` YAML that ``firebase_to_xml`` emits (``identification``, ``spatial``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .. import cf
from ..errors import ProviderError
from ..http import get_json, get_session
from ..query import BBox, Query, register_option
from .base import DiscoverySource, Provider, StationMatch

log = logging.getLogger("omnisea.cioos")

register_option(
    "cioos_records", "cioos_metadata: path, directory, URL or owner/repo holding metadata records"
)
register_option("cioos_token", "cioos_metadata: token for an authenticated records endpoint")

__all__ = ["CioosProvider", "CioosMetadataSource", "parse_record"]

RECORD_SUFFIXES = (".json", ".yaml", ".yml")
GITHUB_SPEC = re.compile(
    r"^(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)(?:@(?P<ref>[\w./-]+))?(?::(?P<path>.+))?$"
)


class CioosProvider(Provider):
    name = "cioos"
    title = "Canadian Integrated Ocean Observing System"
    base_url = "https://form.cioos.ca"
    license = "Per-record; see each record's use_constraints/licence field"
    terms_url = "https://cioos.ca"

    #: Published metadata records are read a few hundred files at a time out of a GitHub
    #: repository, and unauthenticated GitHub allows 60 requests an hour — a day of caching is
    #: the difference between discovery working twice in a row and not.
    cache_policy = {
        "api.github.com/repos/": timedelta(days=1),
        "raw.githubusercontent.com/": timedelta(days=1),
    }

    def build_sources(self) -> Sequence[DiscoverySource]:
        return [CioosMetadataSource(self)]


class CioosMetadataSource(DiscoverySource):
    """Dataset-level metadata records, matched against the query's area and time window."""

    name = "cioos_metadata"
    title = "CIOOS metadata records"
    node_path = "metadata/cioos"
    feature_type = "metadata"

    def discover(self, query: Query) -> list[StationMatch]:
        spec = query.option("cioos_records")
        if not spec:
            # Registered but unconfigured is the normal case: there is no default endpoint, so a
            # plain omnisea.discover() must not fail because of this source.
            log.debug("cioos_metadata: no cioos_records configured; skipping")
            return []

        token = query.option("cioos_token")
        records = load_records(spec, token=token)
        log.debug("cioos_metadata: loaded %d record(s) from %r", len(records), spec)

        matches: list[StationMatch] = []
        for raw in records:
            match = self._match_from_record(query, raw)
            if match is not None:
                matches.append(match.attach_site(query))
        return matches

    def _match_from_record(self, query: Query, raw: Mapping[str, Any]) -> StationMatch | None:
        record = parse_record(raw)
        if record is None:
            return None

        lat, lon = record["lat"], record["lon"]
        if lat is None or lon is None:
            return None
        if not query.contains(lat, lon):
            return None
        if not query.overlaps(record["start"], record["end"]):
            return None

        cf_names = cf.eov_to_cf(record["eov"])
        if query.variables is not None and cf_names:
            wanted = cf.resolve_names(query.variables) or frozenset()
            if not wanted & set(cf_names):
                return None

        return self.new_match(
            station_id=str(record["id"]),
            name=str(record["title"] or record["id"]),
            lat=lat,
            lon=lon,
            variables=cf_names or tuple(record["eov"]),
            n_rows_est=0,  # metadata describes data; it does not deliver rows
            first=_ts(record["start"]),
            last=_ts(record["end"]),
            extra={
                "record": record,
                "eov": record["eov"],
                "distribution": record["distribution"],
                "bbox": record["bbox"],
                "organization": record["organization"],
                "license": record["license"],
                "abstract": record["abstract"],
            },
        )


# --------------------------------------------------------------------------- loading


def load_records(spec: Any, *, token: str | None = None) -> list[dict[str, Any]]:
    """Read metadata records from a path, directory, URL, or ``owner/repo`` GitHub spec."""
    if isinstance(spec, Mapping):
        return _flatten_records(spec)
    if isinstance(spec, (list, tuple)):
        out: list[dict[str, Any]] = []
        for item in spec:
            out.extend(load_records(item, token=token))
        return out

    text = str(spec)
    if text.startswith(("http://", "https://")):
        return _load_url(text, token=token)

    path = Path(os.path.expanduser(text))
    if path.exists():
        return _load_path(path)

    if GITHUB_SPEC.match(text):
        return _load_github(text, token=token)

    raise ProviderError(
        f"could not interpret cioos_records={text!r} as a path, URL or owner/repo spec",
        provider="cioos_metadata",
    )


def _load_path(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        out: list[dict[str, Any]] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in RECORD_SUFFIXES:
                out.extend(_parse_text(child.read_text(encoding="utf-8"), child.name))
        return out
    return _parse_text(path.read_text(encoding="utf-8"), path.name)


def _load_url(url: str, *, token: str | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if token:
        # Firebase RTDB REST reads authenticate with ?auth=<token>.
        params["auth"] = token
    payload = get_json(url, params or None, provider="cioos_metadata")
    return _flatten_records(payload)


def _load_github(spec: str, *, token: str | None) -> list[dict[str, Any]]:
    """Read every record file from a GitHub repository holding published records."""
    m = GITHUB_SPEC.match(spec)
    if m is None:  # pragma: no cover - guarded by the caller
        raise ProviderError(f"bad GitHub spec {spec!r}", provider="cioos_metadata")
    owner, repo = m.group("owner"), m.group("repo")
    ref = m.group("ref")
    subpath = (m.group("path") or "").strip("/")

    session = get_session()
    headers = {"Accept": "application/vnd.github+json"}
    env_token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if env_token:
        headers["Authorization"] = f"Bearer {env_token}"

    if not ref:
        info = session.get(
            f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=(10, 60)
        )
        if not info.ok:
            raise ProviderError(
                f"GitHub repo {owner}/{repo} unreadable (HTTP {info.status_code}); "
                "set cioos_token or GITHUB_TOKEN for a private repo",
                provider="cioos_metadata",
            )
        ref = info.json().get("default_branch", "main")

    tree = session.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}",
        params={"recursive": "1"},
        headers=headers,
        timeout=(10, 120),
    )
    if not tree.ok:
        raise ProviderError(
            f"could not list {owner}/{repo}@{ref} (HTTP {tree.status_code})",
            provider="cioos_metadata",
        )

    files = [
        node["path"]
        for node in tree.json().get("tree", [])
        if node.get("type") == "blob"
        and node["path"].lower().endswith(RECORD_SUFFIXES)
        and (not subpath or node["path"].startswith(subpath))
    ]
    log.debug("cioos_metadata: %d record file(s) in %s/%s@%s", len(files), owner, repo, ref)

    out: list[dict[str, Any]] = []
    for file_path in files:
        raw = session.get(
            f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{file_path}",
            headers={"Authorization": headers["Authorization"]} if env_token else {},
            timeout=(10, 60),
        )
        if raw.ok:
            out.extend(_parse_text(raw.text, file_path))
    return out


def _parse_text(text: str, label: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            return _flatten_records(json.loads(stripped))
        except json.JSONDecodeError:
            pass
    try:
        import yaml  # optional; only needed for YAML records
    except ImportError:
        log.warning("skipping %s: PyYAML is not installed (pip install pyyaml)", label)
        return []
    try:
        return _flatten_records(yaml.safe_load(stripped))
    except Exception:  # noqa: BLE001 - one malformed record must not abort the scan
        log.warning("skipping unparseable record file %s", label, exc_info=True)
        return []


def _flatten_records(payload: Any) -> list[dict[str, Any]]:
    """Pull record dicts out of whatever container they arrived in.

    Firebase exports nest records several levels deep (region -> user -> record id), so this
    walks the structure and keeps anything that looks like a record rather than assuming a shape.
    """
    out: list[dict[str, Any]] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6 or node is None:
            return
        if isinstance(node, list):
            for item in node:
                walk(item, depth + 1)
            return
        if not isinstance(node, Mapping):
            return
        if _looks_like_record(node):
            out.append(dict(node))
            return
        for value in node.values():
            walk(value, depth + 1)

    walk(payload)
    return out


def _looks_like_record(node: Mapping[str, Any]) -> bool:
    if "identification" in node and "spatial" in node:
        return True  # metadata-xml YAML shape
    return "title" in node and ("map" in node or "eov" in node or "distribution" in node)


# --------------------------------------------------------------------------- parsing


def parse_record(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize either record layout into one flat dict.

    Returns ``None`` for a record with no usable spatial extent — a record omnisea cannot place
    on a map cannot answer a spatial query, and guessing a location would be worse than skipping.
    """
    if "identification" in raw and "spatial" in raw:
        return _parse_xml_shape(raw)
    return _parse_form_shape(raw)


def _parse_form_shape(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    extent = raw.get("map") or {}
    bbox = _bbox_from_nsew(extent)
    if bbox is None:
        bbox = _bbox_from_polygon(extent.get("polygon"))
    if bbox is None:
        return None

    return {
        "id": raw.get("recordID") or raw.get("identifier") or raw.get("datasetIdentifier") or "",
        "title": _lang(raw.get("title")),
        "abstract": _lang(raw.get("abstract")),
        "eov": list(raw.get("eov") or []),
        "keywords": list((raw.get("keywords") or {}).get("en") or []),
        "start": raw.get("dateStart"),
        "end": raw.get("dateEnd"),
        "bbox": bbox,
        "lat": bbox.centre[0],
        "lon": bbox.centre[1],
        "distribution": _distribution(raw.get("distribution")),
        "organization": raw.get("organization") or "",
        "license": raw.get("license") or "",
        "progress": raw.get("progress") or raw.get("status") or "",
        "platforms": [p.get("id") for p in raw.get("platforms") or [] if isinstance(p, Mapping)],
    }


def _parse_xml_shape(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    spatial = raw.get("spatial") or {}
    ident = raw.get("identification") or {}
    meta = raw.get("metadata") or {}

    bbox = _bbox_from_list(spatial.get("bbox")) or _bbox_from_polygon(spatial.get("polygon"))
    if bbox is None:
        return None

    keywords = ident.get("keywords") or {}
    eov = keywords.get("eov") or {}

    return {
        "id": ident.get("identifier") or meta.get("identifier") or "",
        "title": _lang(ident.get("title")),
        "abstract": _lang(ident.get("abstract")),
        "eov": list(eov.get("en") or []) if isinstance(eov, Mapping) else list(eov or []),
        "keywords": list((keywords.get("default") or {}).get("en") or []),
        "start": ident.get("temporal_begin") or (ident.get("dates") or {}).get("creation"),
        "end": ident.get("temporal_end"),
        "bbox": bbox,
        "lat": bbox.centre[0],
        "lon": bbox.centre[1],
        "distribution": _distribution(raw.get("distribution")),
        "organization": _first_org(raw.get("contact")),
        "license": ((meta.get("use_constraints") or {}).get("licence") or ""),
        "progress": ident.get("progress_code") or ident.get("status") or "",
        "platforms": [],
    }


def _bbox_from_nsew(extent: Mapping[str, Any]) -> BBox | None:
    try:
        north = float(extent["north"])
        south = float(extent["south"])
        east = float(extent["east"])
        west = float(extent["west"])
    except (KeyError, TypeError, ValueError):
        return None
    return BBox(west, south, east, north)


def _bbox_from_list(bbox: Any) -> BBox | None:
    if not isinstance(bbox, Sequence) or isinstance(bbox, str) or len(bbox) != 4:
        return None
    try:
        west, south, east, north = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return BBox(west, south, east, north)


def _bbox_from_polygon(polygon: Any) -> BBox | None:
    """Bounding box of a ``"lat,lon lat,lon ..."`` polygon string."""
    if not polygon or not isinstance(polygon, str):
        return None
    lats: list[float] = []
    lons: list[float] = []
    for pair in polygon.replace(", ", ",").split():
        parts = pair.split(",")
        if len(parts) != 2:
            continue
        try:
            lats.append(float(parts[0]))
            lons.append(float(parts[1]))
        except ValueError:
            continue
    if not lats:
        return None
    return BBox(min(lons), min(lats), max(lons), max(lats))


def _distribution(dist: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in dist or []:
        if not isinstance(item, Mapping):
            continue
        url = item.get("url") or ""
        if not url:
            continue
        out.append(
            {
                "url": str(url),
                "name": str(_lang(item.get("name")) or ""),
                "description": str(_lang(item.get("description")) or ""),
            }
        )
    return out


def _lang(value: Any, prefer: str = "en") -> str:
    """Bilingual fields are ``{"en": ..., "fr": ...}``; take the preferred language."""
    if isinstance(value, Mapping):
        return str(value.get(prefer) or value.get("fr") or "")
    return "" if value is None else str(value)


def _first_org(contacts: Any) -> str:
    for contact in contacts or []:
        if isinstance(contact, Mapping):
            org = contact.get("organization")
            if isinstance(org, Mapping):
                name = org.get("name")
                if name:
                    return str(name)
    return ""


def _ts(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001 - a malformed record date is not fatal
        return None
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
