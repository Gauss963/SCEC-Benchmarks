#!/usr/bin/env python3
"""Download the official CVWS-generated TPV101 GIF plots for public users."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from fetch_scec_reference import (
    PUBLIC_USERS,
    STATION_LIST,
    _open_file_list,
    _post,
)
from tpv101 import STATIONS


def _image_urls(page: str) -> list[str]:
    urls = re.findall(r'src="([^"]*cvws\.gif\?[^"]+)"', page, flags=re.IGNORECASE)
    return [html.unescape(url) for url in urls]


def _download_gif(
    opener: urllib.request.OpenerDirector, url: str, path: Path, retries: int = 3
) -> None:
    if path.exists() and path.stat().st_size > 100 and path.read_bytes()[:3] == b"GIF":
        return
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with opener.open(url, timeout=90.0) as response:
                payload = response.read()
            if not payload.startswith((b"GIF87a", b"GIF89a")):
                raise RuntimeError(f"CVWS did not return a GIF for {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return
        except Exception as error:  # pragma: no cover - network retry path
            last_error = error
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}") from last_error


def fetch_user_plots(user: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    manifest = []
    for station_name, _x, _y in STATIONS:
        opener = _open_file_list(user)
        page = _post(
            opener,
            {
                f"G1068{station_name}": "  Graph  ",
                "m": "tpv101",
                "o": "1005",
                "mus": user,
                "Q0001": STATION_LIST,
            },
        )
        urls = _image_urls(page)
        if len(urls) != 8:
            raise RuntimeError(
                f"Expected 8 station plots for {user}/{station_name}, got {len(urls)}"
            )
        for url in urls:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            field = query.get("cvf", ["unknown"])[0]
            path = output_dir / station_name / f"{field}.gif"
            _download_gif(opener, url, path)
            paths.append(path)
            manifest.append({"path": str(path), "source_url": url})
        print(f"Downloaded official plots for {user}/{station_name}", flush=True)

    opener = _open_file_list(user)
    page = _post(
        opener,
        {
            "G1078cplot": " Graph ",
            "m": "tpv101",
            "o": "1005",
            "mus": user,
            "Q0001": STATION_LIST,
        },
    )
    urls = _image_urls(page)
    if len(urls) != 1:
        raise RuntimeError(f"Expected one contour plot for {user}, got {len(urls)}")
    contour_path = output_dir / "rupture_time_contour.gif"
    _download_gif(opener, urls[0], contour_path)
    paths.append(contour_path)
    manifest.append({"path": str(contour_path), "source_url": urls[0]})
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "TPV101",
                "user": user,
                "source": "https://strike.scec.org/cvws/cgi-bin/cvws.cgi",
                "plots": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Downloaded official contour for {user}", flush=True)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", choices=(*PUBLIC_USERS, "all"), default="all")
    parser.add_argument("--reference-root", type=Path, default=Path("reference"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    users = PUBLIC_USERS if args.user == "all" else (args.user,)
    total = 0
    for user in users:
        paths = fetch_user_plots(
            user, args.reference_root / f"{user}_100m" / "site_plots"
        )
        total += len(paths)
    print(f"Downloaded or verified {total} official CVWS plots", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
