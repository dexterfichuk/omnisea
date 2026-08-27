"""ERDDAP — one adapter for the tens of thousands of datasets served by ERDDAP installations.

ERDDAP is the same software running at IOOS, CIOOS, NOAA CoastWatch, EMODnet and a few hundred
other institutions, and every installation answers the same handful of URLs. That is why one
adapter is the largest coverage win available: point it at a different ``erddap_server`` and the
whole catalogue of that institution becomes queryable with no new code.

Two protocols, two sources, one seam:

* ``tabledap`` (:mod:`.table`) serves station and platform records as tables, so
  :class:`ErddapTableSource` reuses omnisea's point path and returns
  :class:`~omnisea.providers.base.StationSeries`.
* ``griddap`` (:mod:`.grid`) serves gridded fields over OPeNDAP, so :class:`ErddapGridSource`
  returns a **lazy** :class:`xarray.Dataset` tagged with ``omnisea_node_path``. Nothing is read
  until the caller actually indexes it.

What they share — discovery over ``allDatasets`` plus the search index, the ``/info`` metadata
reads, the 404-means-empty convention — lives in :mod:`.common`; the metadata model itself is
:mod:`.info`.

**No field table is hardcoded here.** Every ERDDAP dataset publishes its own CF metadata —
``standard_name``, ``units``, ``cell_methods``, ``ancillary_variables`` — through
``/info/{dataset_id}/index.json``, and this adapter reads that and passes it through. Inventing a
mapping for a catalogue this size would be both impossible and wrong: the dataset author knows
what they measured. The consequence is that :attr:`ErddapSource.fields` is empty at class level
and the real table is built per dataset at fetch time.

Two things that follow from reading the metadata rather than curating it, both worth knowing:

* **``to_cf_units=True`` cannot convert anything here.** A dataset states the units its numbers
  are in; it does not state how to reach canonical CF units, and omnisea will not guess a scale
  factor for someone else's data. Values therefore always come back in the published units, with
  those units in the ``units`` attribute — which is the invariant that matters.
* **tabledap nodes go under ``in_situ/``.** Every ``cdm_data_type`` tabledap serves is a CF
  discrete-sampling geometry, which is an observation by definition, so this holds for the
  datasets that exist today. It is an inference and not a guarantee, so ``cdm_data_type`` and the
  dataset's own ``source_url`` are recorded on every node for anyone who needs to check.

**Why not erddapy.** ``erddapy`` builds these URLs and hands back a DataFrame, which is the easy
half. The hard half is routing through :mod:`omnisea.http` so that the retry policy, the global
concurrency cap, the User-Agent and the payload ceiling apply — and erddapy reads with its own
``pandas.read_csv`` call, outside all of that. Re-implementing the URL strings is a few lines;
re-implementing the safety is not. So this package talks to the REST endpoints directly and the
``erddap`` extra stays optional and unused.

Three upstream behaviours shape the code, all verified live:

* **Zero results are HTTP 404.** ``search/advanced.json``, ``allDatasets`` and ``tabledap`` all
  answer an empty result set with ``404 ... produced no matching results``. That is not a
  failure, so it is translated to "nothing here" — while a 404 saying ``Currently unknown
  datasetID`` still raises, because that one really is wrong.
* **``allDatasets`` is not always populated.** CIOOS Pacific publishes null bounds for every row
  of ``allDatasets`` while its search index knows the bounds perfectly well; IOOS Sensors fills
  both. Neither endpoint alone lists every dataset in a box, so both are consulted and the ids
  unioned rather than trusting whichever answers first.
* **ERDDAP publishes no row count.** There is no cheap "how many rows would this be?" call, so
  the estimate comes from ``time_coverage_resolution`` where the dataset declares it and from a
  documented assumption where it does not — and the ceiling is additionally enforced against the
  rows actually returned, so a bad estimate cannot become an unbounded download.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..base import Provider, RetrievalSource
from .common import DEFAULT_MAX_DATASETS, DEFAULT_SERVER, ErddapSource, safe_name, table_rows
from .grid import ErddapGridSource, grid_selection
from .info import (
    DEFAULT_SAMPLES_PER_DAY,
    DatasetInfo,
    clear_cache,
    parse_info,
)
from .servers import SERVERS, ErddapServer
from .table import ROWS_PER_REQUEST, ErddapTableSource, field_table

__all__ = [
    "NAMED_PROVIDERS",
    "ErddapServer",
    "ErddapProvider",
    "ErddapSource",
    "ErddapTableSource",
    "ErddapGridSource",
    "DatasetInfo",
    "parse_info",
    "clear_cache",
    "field_table",
    "grid_selection",
    "table_rows",
    "safe_name",
    "DEFAULT_SERVER",
    "DEFAULT_MAX_DATASETS",
    "DEFAULT_SAMPLES_PER_DAY",
    "ROWS_PER_REQUEST",
]


class ErddapProvider(Provider):
    """One ERDDAP installation.

    ``base_url`` is only the default: an ERDDAP server is interchangeable with any other, so the
    server actually queried comes from ``erddap_server=`` and is recorded on every node produced,
    together with the dataset's own licence — which on ERDDAP is per dataset, not per server.
    """

    name = "erddap"
    title = "ERDDAP"
    base_url = DEFAULT_SERVER
    license = "Per-dataset; see each dataset's 'license' global attribute"
    #: No provider-level terms URL. ERDDAP is software, not a publisher: one installation hosts
    #: a dozen institutions under a dozen terms, and the placeholder here used to be NOAA
    #: PFEG's ERDDAP *information page* -- which citation() then printed under DFO, Hakai, NDBC
    #: and GHRSST rows alike, sending anyone following the link somewhere unrelated. Where the
    #: dataset states its own, _node_attrs uses that; where it does not, no link is better than
    #: a wrong one.
    terms_url = ""

    #: The installation this provider *is*, when it is one. The generic adapter leaves this
    #: unset and takes its server from ``erddap_server=``; a named one pins it, so asking for
    #: ``providers="hakai"`` cannot be silently redirected somewhere else by an option.
    server: ErddapServer | None = None

    def clear_cache(self) -> None:
        clear_cache()

    def build_sources(self) -> Sequence[RetrievalSource]:
        return [ErddapTableSource(self), ErddapGridSource(self)]


def _named_provider(server: ErddapServer) -> type[ErddapProvider]:
    """Build the Provider class for one known installation.

    ERDDAP is software; Hakai, IOOS and NOAA CoastWatch are organizations. Modelling the
    software as the provider put "erddap" in the organization column for CO-OPS, NDBC and
    NESDIS alike, and left the eleven installations omnisea knows reachable only through an
    option nobody discovers. One provider per installation makes them ordinary: they appear in
    ``omnisea.sources()``, ``providers="hakai"`` selects one the way ``providers="eccc"`` does,
    and an unqualified query sweeps the ones whose region contains it.
    """

    class _Named(ErddapProvider):
        name = server.name
        title = server.institution or server.name
        base_url = server.url
        # The dataset's own licence still wins on every node; this is the fallback for a
        # dataset that states none, and it names the host rather than the software.
        license = f"Per-dataset; published via {server.institution or server.name}"

    _Named.server = server
    _Named.__name__ = "".join(p.title() for p in server.name.split("_")) + "ErddapProvider"
    _Named.__qualname__ = _Named.__name__
    _Named.__doc__ = f"{server.institution or server.name} — {server.holdings}."
    return _Named


#: One provider per known installation, built from the registry so the two cannot drift.
NAMED_PROVIDERS: dict[str, type[ErddapProvider]] = {
    name: _named_provider(server) for name, server in SERVERS.items()
}
