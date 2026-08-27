"""Known ERDDAP installations, so reaching one does not require knowing its URL.

One adapter reads every ERDDAP, which makes the remaining barrier social rather than technical:
you cannot query a server you have never heard of. These are the installations omnisea has been
checked against, each reduced to a short name — ``erddap_server="hakai"`` instead of a URL
somebody has to find and paste correctly.

The list is a **convenience, not a boundary**. ``erddap_server=`` still takes any URL, and an
installation absent from this table is not second-class; there are several hundred ERDDAPs and
curating all of them would be a full-time job with a stale result. What is here is what has been
confirmed to answer, with the dataset counts observed when it was.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ErddapServer",
    "SERVERS",
    "resolve_servers",
    "server_name_for_url",
    "server_table",
]


@dataclass(frozen=True)
class ErddapServer:
    """One ERDDAP installation: where it is, whose it is, and what it is good for."""

    #: Short name accepted by ``erddap_server=``. Also the node-path segment, so it must be a
    #: bare identifier.
    name: str
    url: str
    institution: str
    #: What the catalogue is worth querying *for*, in a phrase. Shown by
    #: :func:`omnisea.erddap_servers`, which is where someone decides whether to look.
    holdings: str
    #: Roughly how many datasets it published when it was last checked. An order of magnitude,
    #: not a promise — it says whether a server is a handful of moorings or a national archive.
    datasets: int = 0
    #: Where this installation actually has data, as ``(west, south, east, north)``, or ``None``
    #: for a global one. A regional server is skipped for a query outside its region rather than
    #: asked a question it cannot answer — SalishSeaCast has nothing to say about Nova Scotia.
    #: Drawn generously: it decides what an *unqualified* query sweeps, and naming the provider
    #: reaches it regardless of where you are asking about.
    coverage: tuple[float, float, float, float] | None = None
    #: Whether an unqualified ``discover()`` sweeps this installation. False for the global
    #: satellite archives: they match essentially any bounding box, so every bare query would
    #: hit the dataset ceiling on them and carry a ceiling note. Reached by name instead.
    sweep: bool = True


#: Keyed by short name. Ordered roughly by how likely a Northeast Pacific query is to want them,
#: which is also the order :func:`omnisea.erddap_servers` prints.
SERVERS: dict[str, ErddapServer] = {
    server.name: server
    for server in (
        ErddapServer(
            "ioos_sensors",
            "https://erddap.sensors.ioos.us/erddap",
            "US Integrated Ocean Observing System",
            "the largest aggregation of US coastal sensors — NDBC buoys, tide gauges, "
            "met stations",
            27_260,
        ),
        ErddapServer(
            "cioos_pacific",
            "https://data.cioospacific.ca/erddap",
            "Fisheries and Oceans Canada / CIOOS Pacific",
            "DFO Pacific's long-term programs: Line P, C-PROOF gliders, BC lighthouse "
            "records, cruise CTD and bottle data",
            122,
            coverage=(-146.0, 46.0, -120.0, 56.0),
        ),
        ErddapServer(
            "hakai",
            "https://catalogue.hakai.org/erddap",
            "Hakai Institute",
            "central BC coast moorings, profiles and nearshore observatories",
            63,
            coverage=(-132.0, 48.0, -122.0, 55.0),
        ),
        ErddapServer(
            "salishseacast",
            "https://salishsea.eos.ubc.ca/erddap",
            "University of British Columbia",
            "the SalishSeaCast NEMO model — hourly 3-D physics and biology for the Salish Sea",
            53,
            coverage=(-126.5, 46.8, -121.5, 51.0),
        ),
        ErddapServer(
            "nwem",
            "https://nwem.apl.uw.edu/erddap",
            "University of Washington Applied Physics Laboratory",
            "ORCA and NEMO moorings through Puget Sound and the Washington coast",
            122,
            coverage=(-126.0, 46.0, -121.5, 49.5),
        ),
        ErddapServer(
            "coastwatch",
            "https://coastwatch.noaa.gov/erddap",
            "NOAA CoastWatch",
            "global satellite fields — SST, ocean colour, winds, sea ice",
            454,
            sweep=False,  # matches any bbox; would hit the dataset ceiling every time
        ),
        ErddapServer(
            "coastwatch_west",
            "https://coastwatch.pfeg.noaa.gov/erddap",
            "NOAA CoastWatch West Coast Node",
            "satellite and model grids with a US west coast emphasis, plus long SST records",
            3_053,
            sweep=False,
        ),
        ErddapServer(
            "glider_dac",
            "https://gliders.ioos.us/erddap",
            "US IOOS Glider Data Assembly Center",
            "glider deployments worldwide, including DFO's",
            2_542,
        ),
        ErddapServer(
            "osmc",
            "https://osmc.noaa.gov/erddap",
            "NOAA Observing System Monitoring Center",
            "the global in-situ feed — Argo floats, drifters, ships, moored buoys, tide gauges",
            30,
        ),
        ErddapServer(
            "cioos_atlantic",
            "https://cioosatlantic.ca/erddap",
            "CIOOS Atlantic",
            "Atlantic Canada moorings, gliders and coastal monitoring",
            186,
            coverage=(-70.0, 40.0, -47.0, 56.0),
        ),
        ErddapServer(
            "cioos_slgo",
            "https://erddap.ogsl.ca/erddap",
            "CIOOS St. Lawrence / Observatoire global du Saint-Laurent",
            "Gulf and Estuary of St. Lawrence observations",
            96,
            coverage=(-72.0, 44.0, -55.0, 53.0),
        ),
    )
}


def resolve_servers(spec: object, default: str) -> list[ErddapServer]:
    """Turn an ``erddap_server=`` value into the servers to query.

    Accepts a short name, a full URL, a list of either, or ``"all"`` for every known
    installation. ``None`` gives the source's own default. A URL that is not in the table is
    returned as an unnamed server rather than refused — the registry is a convenience, and
    refusing an unlisted ERDDAP would make it a whitelist.
    """
    from ...errors import QueryError

    if spec is None or spec == "":
        spec = default
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return list(SERVERS.values())

    items = [spec] if isinstance(spec, str) else list(spec)  # type: ignore[arg-type]
    if not items:
        raise QueryError("erddap_server=[] names no server to query")

    out: list[ErddapServer] = []
    seen: set[str] = set()
    for item in items:
        server = _one(str(item).strip())
        if server.url not in seen:
            seen.add(server.url)
            out.append(server)
    return out


def _one(item: str) -> ErddapServer:
    from ...errors import QueryError

    if item in SERVERS:
        return SERVERS[item]
    if item.startswith(("http://", "https://")):
        url = item.rstrip("/")
        for known in SERVERS.values():
            if known.url == url:
                return known
        return ErddapServer(_name_from_url(url), url, "", "")
    known = ", ".join(sorted(SERVERS))
    raise QueryError(
        f"erddap_server={item!r} is neither a known server name nor a full http(s) URL "
        f"ending at the ERDDAP root. Known names: {known}. Pass 'all' to query every one, "
        "or omnisea.erddap_servers() to see what each holds."
    )


def server_name_for_url(url: str) -> str:
    """The short name for a server URL — the registry's if it is listed, else from its host.

    Used for node paths, so that where a dataset lands does not depend on whether discovery
    happened to record the name.
    """
    cleaned = url.rstrip("/")
    for known in SERVERS.values():
        if known.url == cleaned:
            return known.name
    return _name_from_url(cleaned)


def _name_from_url(url: str) -> str:
    """A node-path segment for a server that is not in the table, from its host."""
    host = url.split("//", 1)[-1].split("/", 1)[0]
    cleaned = "".join(char if char.isalnum() else "_" for char in host.removeprefix("www."))
    return cleaned.strip("_").lower() or "erddap"


def server_table() -> list[dict[str, object]]:
    """The registry as rows, for :func:`omnisea.erddap_servers`."""
    return [
        {
            "server": s.name,
            "institution": s.institution,
            "holdings": s.holdings,
            "datasets": s.datasets,
            "url": s.url,
        }
        for s in SERVERS.values()
    ]
