#!/usr/bin/env python3
"""Download public TPV101/TPV102 raw files from the SCEC comparison server."""

from __future__ import annotations

import argparse
import hashlib
import html
import http.cookiejar
import re
import urllib.parse
import urllib.request
from pathlib import Path

from tpv101 import STATIONS, SURFACE_STATIONS


CVWS_URL = "https://strike.scec.org/cvws/cgi-bin/cvws.cgi"
PUBLIC_USERS = ("barall", "dalguer2", "dunham", "kaneko", "ke", "liu")
PUBLIC_IDENTITIES = {
    "barall": "Michael Barall - Finite Element - FaultMod",
    "dalguer2": "Luis Dalguer - Finite Difference - DFM",
    "dunham": "Eric Dunham - Boundary Integral - MDSBI",
    "kaneko": "Yoshihiro Kaneko - Spectral Element - SPECFEM3D",
    "ke": "Chun-Yu Ke - Spectral Boundary Integral - uguca",
    "liu": "Yi Liu - Boundary Integral",
}
USER_LIST = "*".join(PUBLIC_USERS)
TPV102_USERS = (
    "aagaard",
    "aagaard.2",
    "aagaard.3",
    "aagaard.4",
    "barall",
    "dalguer2",
    "kaneko",
    "ke",
    "liu",
    "luo",
    "ma",
    "ma.2",
    "wzhang",
)
USER_LISTS = {"tpv101": USER_LIST, "tpv102": "*".join(TPV102_USERS)}
STATION_LIST = "*".join(station[0] for station in STATIONS)


def _post(opener: urllib.request.OpenerDirector, values: dict[str, str]) -> str:
    request = urllib.request.Request(
        CVWS_URL,
        data=urllib.parse.urlencode(values).encode("ascii"),
    )
    with opener.open(request, timeout=60.0) as response:
        return response.read().decode("utf-8")


def _extract_raw_data(page: str) -> str:
    match = re.search(r"<pre>(.*?)</pre>", page, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        title = re.search(r"<title>(.*?)</title>", page, flags=re.IGNORECASE)
        page_title = html.unescape(title.group(1)) if title else "unknown page"
        raise RuntimeError(f"SCEC server did not return raw data ({page_title}).")
    return html.unescape(match.group(1)).strip() + "\n"


def _open_file_list(
    user: str, benchmark: str = "tpv101", *, include_page: bool = False
) -> urllib.request.OpenerDirector | tuple[urllib.request.OpenerDirector, str]:
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    _post(opener, {"G0012": "Go -->", "o": "1005"})
    _post(opener, {f"G1045{benchmark}": " Select ", "o": "1005"})
    page = _post(
        opener,
        {
            f"G1047{user}": " Select ",
            "m": benchmark,
            "o": "1005",
            "Q0001": USER_LISTS[benchmark],
        },
    )
    if "Select File(s)" not in page:
        raise RuntimeError(f"Could not open the {benchmark.upper()} file list for user {user}.")
    return (opener, page) if include_page else opener


def fetch_reference(
    user: str, output_dir: Path, *, benchmark: str = "tpv101"
) -> list[Path]:
    if benchmark not in USER_LISTS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if user not in PUBLIC_IDENTITIES:
        raise ValueError(f"Unknown public user: {user}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    _opener, file_list_page = _open_file_list(user, benchmark, include_page=True)
    available_raw_files = set(
        re.findall(r'name="G1059([^"]+)"', file_list_page, flags=re.IGNORECASE)
    )
    stations = list(STATIONS)
    if benchmark == "tpv102":
        stations.extend(
            station for station in SURFACE_STATIONS if station[0] in available_raw_files
        )

    for station_name, _x, _y in stations:
        opener = _open_file_list(user, benchmark)
        page = _post(
            opener,
            {
                f"G1059{station_name}": "Raw Data",
                "m": benchmark,
                "o": "1005",
                "mus": user,
                "Q0001": STATION_LIST,
            },
        )
        path = output_dir / f"{station_name}.txt"
        path.write_text(_extract_raw_data(page), encoding="ascii")
        paths.append(path)
        print(f"Downloaded {user}/{path.name}")

    opener = _open_file_list(user, benchmark)
    page = _post(
        opener,
        {
            "G1063cplot": "Raw Data",
            "m": benchmark,
            "o": "1005",
            "mus": user,
            "Q0001": STATION_LIST,
        },
    )
    contour_path = output_dir / f"{benchmark}_rupture_time.txt"
    contour_path.write_text(_extract_raw_data(page), encoding="ascii")
    paths.append(contour_path)
    print(f"Downloaded {user}/{contour_path.name}")

    official_listing = PUBLIC_IDENTITIES[user]
    source_path = output_dir / "SOURCE.txt"
    source_path.write_text(
        "SCEC/USGS Spontaneous Rupture Code Verification Project\n"
        f"Benchmark: {benchmark.upper()}\nUser: {user}\n"
        f"Official listing: {official_listing}\n"
        f"Source: {CVWS_URL}\n",
        encoding="ascii",
    )
    paths.append(source_path)

    checksum_path = output_dir / "CHECKSUMS.sha256"
    checksum_lines = []
    for path in paths:
        if path == source_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {path.name}")
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    paths.append(checksum_path)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", choices=PUBLIC_USERS, default="dunham")
    parser.add_argument("--benchmark", choices=tuple(USER_LISTS), default="tpv101")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (
        Path("reference") / f"{args.user}_100m"
        if args.benchmark == "tpv101"
        else Path("reference") / args.benchmark / f"{args.user}_100m"
    )
    fetch_reference(args.user, output_dir, benchmark=args.benchmark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
