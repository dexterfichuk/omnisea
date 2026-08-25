"""Where the data came from — attribution you can cite, derived from the result itself.

Every node omnisea builds already carries the institution that published it, the licence it
came under and the URL it was read from. This module reads that back out, so the question
"what do I cite?" is answered by the data rather than by remembering what you ran.

That matters more than it sounds. A single tree can hold observations from four organizations
under three licences, some of it realtime and therefore unreproducible after the fact. Writing
that up correctly from memory is exactly the sort of thing that gets skipped.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import xarray as xr

from .tree import data_nodes, scalar_coord

__all__ = ["provenance", "citation", "sources_used"]


def provenance(tree: xr.DataTree, *, by: str = "source") -> pd.DataFrame:
    """One row per data source in the tree, with everything needed to attribute it.

    ``by="source"`` groups by dataset (``eccc_climate_daily``), ``by="provider"`` by
    organization (``eccc``), ``by="node"`` gives the full per-station detail including the exact
    URL each series was read from.
    """
    rows: list[dict[str, Any]] = []
    for path, ds in data_nodes(tree):
        attrs = ds.attrs
        times = ds["time"].values if "time" in ds.coords and ds["time"].size else None
        rows.append(
            {
                "provider": attrs.get("provider", ""),
                "source": attrs.get("source_name", ""),
                "institution": attrs.get("institution", ""),
                "license": attrs.get("license", ""),
                "terms": attrs.get("references", ""),
                "collection": attrs.get("collection", ""),
                "station_id": scalar_coord(ds, "station_id"),
                "station_name": scalar_coord(ds, "station_name"),
                "node": path,
                "source_url": attrs.get("source_url", ""),
                "n_time": int(times.size) if times is not None else 0,
                "first": pd.Timestamp(times.min()) if times is not None else pd.NaT,
                "last": pd.Timestamp(times.max()) if times is not None else pd.NaT,
                "variables": ", ".join(
                    str(v) for v in ds.data_vars if not str(v).endswith("_qc")
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty or by == "node":
        return frame.sort_values(["provider", "source", "node"]).reset_index(drop=True) \
            if not frame.empty else frame

    if by not in ("source", "provider"):
        raise ValueError(f"by must be 'source', 'provider' or 'node'; got {by!r}")

    keys = ["provider", "source"] if by == "source" else ["provider"]
    # Attribution joins the grouping keys rather than being aggregated with "first": one ERDDAP
    # source can serve datasets from several institutions under several licences, and collapsing
    # them onto whichever row came first would mis-cite every other one. Sources with uniform
    # attribution — which is all of the non-ERDDAP ones — still come out as one row each.
    keys += ["institution", "license", "terms"]
    grouped = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            n_stations=("station_id", "nunique"),
            n_nodes=("node", "count"),
            n_values=("n_time", "sum"),
            first=("first", "min"),
            last=("last", "max"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return grouped


def sources_used(tree: xr.DataTree) -> list[str]:
    """Just the source names present, for a quick check."""
    return sorted({ds.attrs.get("source_name", "") for _, ds in data_nodes(tree)} - {""})


def citation(
    tree: xr.DataTree,
    *,
    style: str = "text",
    include_urls: bool = False,
) -> str:
    """An attribution block for a methods section, built from what was actually retrieved.

    ``style="text"`` is prose meant to be pasted and lightly edited; ``style="markdown"`` is the
    same content as a list. Pass ``include_urls=True`` to list the exact endpoint each station's
    series was read from — worth doing for realtime sources, whose contents cannot be recovered
    later from the query alone.
    """
    if style not in ("text", "markdown"):
        raise ValueError(f"style must be 'text' or 'markdown'; got {style!r}")

    detail = provenance(tree, by="node")
    if detail.empty:
        # An empty tree is the case where saying *why* matters most: "no data" and "a server
        # was down" look identical here, and only one of them means there is nothing to study.
        reasons = _incompleteness_lines(tree, "- " if style == "markdown" else "  ")
        if not reasons:
            return "No data sources — this tree is empty."
        return "\n".join(["No data was retrieved. Why:", "", *reasons])

    summary = provenance(tree, by="source")
    accessed = str(tree.attrs.get("history", "")).split(" ", 1)[0] or "unknown"
    version = tree.attrs.get("omnisea_version", "")

    window = _describe_window(detail)
    lines: list[str] = []
    bullet = "- " if style == "markdown" else "  "

    # Counted from the per-node detail, not from the summary rows: a source serving datasets
    # from two institutions produces two attribution entries but is still one source.
    header = (
        f"Data were retrieved with omnisea {version} on {accessed} "
        f"from {detail['source'].nunique()} source(s) across "
        f"{detail['station_id'].nunique()} station(s){window}."
    )
    lines.append(f"**Data sources**\n\n{header}" if style == "markdown" else header)
    lines.append("")

    for _, row in summary.iterrows():
        span = _describe_span(row["first"], row["last"])
        entry = (
            f"{row['institution'] or row['provider']} — {row['source']}"
            f" ({row['n_stations']} station(s){span}). "
            f"Licence: {row['license'] or 'see provider'}."
        )
        if row["terms"]:
            entry += f" Terms: {row['terms']}"
        lines.append(f"{bullet}{entry}")

        if include_urls:
            # Filtered on the attribution too, so a split source lists each institution's
            # stations under its own entry rather than everything under both.
            urls = detail.loc[
                (detail["source"] == row["source"])
                & (detail["institution"] == row["institution"]),
                ["station_id", "source_url"],
            ]
            for _, url_row in urls.iterrows():
                if url_row["source_url"]:
                    prefix = "    - " if style == "markdown" else "      "
                    lines.append(f"{prefix}{url_row['station_id']}: {url_row['source_url']}")

    caveats = _incompleteness_lines(tree, bullet)
    if caveats:
        lines.append("")
        lines.extend(caveats)

    return "\n".join(lines)


def _incompleteness_lines(tree: xr.DataTree, bullet: str) -> list[str]:
    """Everything that qualifies the result: failures, empty stations, coverage gaps.

    Shared by the populated and the empty citation, so a retrieval that returned nothing still
    explains itself instead of reading as "there is no data here".
    """
    lines: list[str] = []

    incomplete = tree.attrs.get("omnisea_fetch_errors")
    if incomplete:
        lines.append(
            f"{bullet}NOTE: this retrieval was incomplete. Failed sources: {incomplete}"
        )

    empty = tree.attrs.get("omnisea_empty_stations")
    if empty:
        listed = empty if isinstance(empty, str) else ", ".join(map(str, empty))
        lines.append(
            f"{bullet}NOTE: these stations were matched but returned no rows: {listed}"
        )

    notes = tree.attrs.get("omnisea_source_notes")
    if notes:
        lines.append(
            f"{bullet}NOTE: a source could not cover the requested window: {notes}"
        )

    return lines


def _describe_window(detail: pd.DataFrame) -> str:
    first, last = detail["first"].min(), detail["last"].max()
    if pd.isna(first) or pd.isna(last):
        return ""
    return f", covering {first:%Y-%m-%d} to {last:%Y-%m-%d}"


def _describe_span(first: Any, last: Any) -> str:
    if pd.isna(first) or pd.isna(last):
        return ""
    return f", {pd.Timestamp(first):%Y-%m-%d} to {pd.Timestamp(last):%Y-%m-%d}"
