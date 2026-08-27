#!/usr/bin/env python3
"""Check a SCEC TPV101/TPV102 upload set against uploadTPV101.pdf."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ON_FAULT_FIELDS = (
    "t h-slip h-slip-rate h-shear-stress v-slip v-slip-rate "
    "v-shear-stress n-stress log-theta"
).split()
OFF_FAULT_FIELDS = "t h-disp h-vel v-disp v-vel n-disp n-vel".split()
CONTOUR_FIELDS = ["j", "k", "t"]
REQUIRED_KEYS = (
    "problem", "author", "date", "code", "element_size",
    "time_step", "num_time_steps", "location",
)
ON_FAULT_STATIONS = {
    "faultst-120dp030": (-12.0, 3.0), "faultst000dp030": (0.0, 3.0),
    "faultst120dp030": (12.0, 3.0), "faultst-090dp075": (-9.0, 7.5),
    "faultst000dp075": (0.0, 7.5), "faultst090dp075": (9.0, 7.5),
    "faultst-120dp120": (-12.0, 12.0), "faultst000dp120": (0.0, 12.0),
    "faultst120dp120": (12.0, 12.0),
}
OFF_FAULT_STATIONS = {
    "body-060st-120dp000", "body-090st000dp000", "body-060st120dp000",
    "body060st-120dp000", "body090st000dp000", "body060st120dp000",
}

problems: list[str] = []
notes: list[str] = []


def fail(message: str) -> None:
    problems.append(message)


def parse(path: Path) -> tuple[dict[str, str], list[str], np.ndarray]:
    header: dict[str, str] = {}
    fields: list[str] | None = None
    rows: list[list[float]] = []
    columns_documented = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                body = stripped.lstrip("#").strip()
                if body.startswith("Column #"):
                    columns_documented += 1
                elif "=" in body:
                    key, _, value = body.partition("=")
                    header[key.strip()] = value.strip()
                continue
            try:
                rows.append([float(value) for value in stripped.split()])
            except ValueError:
                if fields is None:
                    fields = stripped.split()
                else:
                    fail(f"{path.name}: unparsable line {stripped[:40]!r}")
    header["_columns_documented"] = str(columns_documented)
    return header, fields or [], np.asarray(rows, dtype=float)


def check_header(path: Path, header: dict[str, str], problem: str) -> None:
    for key in REQUIRED_KEYS:
        if key not in header:
            fail(f"{path.name}: header is missing '{key}='")
    if header.get("problem") != problem:
        fail(f"{path.name}: problem={header.get('problem')!r}, expected {problem!r}")
    author = header.get("author", "")
    if not author or "Modeler" in author or "workflow" in author:
        fail(f"{path.name}: author looks like a placeholder: {author!r}")


def check_series(path: Path, problem: str, fields_expected: list[str]) -> None:
    header, fields, data = parse(path)
    check_header(path, header, problem)
    if fields != fields_expected:
        fail(f"{path.name}: field list is {' '.join(fields)!r}")
    documented = int(header["_columns_documented"])
    if documented != len(fields_expected):
        fail(f"{path.name}: {documented} 'Column #' lines, expected {len(fields_expected)}")
    if data.ndim != 2 or data.shape[1] != len(fields_expected):
        fail(f"{path.name}: data shape {data.shape}, expected N x {len(fields_expected)}")
        return
    declared = int(header.get("num_time_steps", -1))
    if declared != data.shape[0]:
        fail(f"{path.name}: num_time_steps={declared} but {data.shape[0]} rows")
    times = data[:, 0]
    if not np.all(np.diff(times) > 0):
        fail(f"{path.name}: time column is not strictly increasing")
    steps = np.diff(times)
    if steps.size and float(np.max(np.abs(steps - steps[0]))) > 1.0e-9:
        fail(f"{path.name}: time step is not uniform")
    declared_step = float(header.get("time_step", "nan"))
    if steps.size and not math.isclose(declared_step, float(steps[0]), rel_tol=1e-9):
        fail(f"{path.name}: time_step={declared_step} but spacing is {steps[0]}")
    if not np.all(np.isfinite(data)):
        fail(f"{path.name}: contains non-finite values")
    location = header.get("location", "")
    if fields_expected is ON_FAULT_FIELDS:
        strike, dip = ON_FAULT_STATIONS[path.stem]
        for token in (f"{strike:g} km along strike", f"{dip:g} km down-dip"):
            if token not in location:
                fail(f"{path.name}: location {location!r} does not mention {token!r}")
        first = data[0]
        if not math.isclose(first[3], 75.0, abs_tol=1e-3):
            fail(f"{path.name}: initial h-shear-stress is {first[3]}, expected 75 MPa")
        if not math.isclose(first[7], 120.0, abs_tol=1e-3):
            fail(f"{path.name}: initial n-stress is {first[7]}, expected 120 MPa")
        if abs(first[1]) > 1e-12 or abs(first[4]) > 1e-12:
            fail(f"{path.name}: initial slip is not zero")
    return


def check_contour(path: Path, problem: str) -> None:
    header, fields, data = parse(path)
    for key in ("problem", "author", "date", "code", "element_size"):
        if key not in header:
            fail(f"{path.name}: header is missing '{key}='")
    if header.get("problem") != problem:
        fail(f"{path.name}: problem={header.get('problem')!r}, expected {problem!r}")
    if fields != CONTOUR_FIELDS:
        fail(f"{path.name}: field list is {' '.join(fields)!r}, expected 'j k t'")
    if data.shape[1] != 3:
        fail(f"{path.name}: {data.shape[1]} columns, expected 3")
        return
    j, k, t = data[:, 0], data[:, 1], data[:, 2]
    if j.min() < -15000.0 - 1e-6 or j.max() > 15000.0 + 1e-6:
        fail(f"{path.name}: j outside [-15000, 15000]: {j.min()} to {j.max()}")
    if k.min() < -1e-6 or k.max() > 15000.0 + 1e-6:
        fail(f"{path.name}: k outside [0, 15000]: {k.min()} to {k.max()}")
    unruptured = t >= 1.0e9 - 1.0
    if unruptured.any() and not np.allclose(t[unruptured], 1.0e9):
        fail(f"{path.name}: unruptured nodes are not exactly 1.0E+09")
    finite = t[~unruptured]
    if finite.size and (finite.min() < 0.0):
        fail(f"{path.name}: negative rupture time")
    if np.unique(data[:, :2], axis=0).shape[0] != data.shape[0]:
        fail(f"{path.name}: duplicate (j, k) nodes")
    notes.append(
        f"{path.name}: {data.shape[0]} nodes, j {j.min():.0f}..{j.max():.0f} m, "
        f"k {k.min():.0f}..{k.max():.0f} m, rupture {finite.min():.4f}..{finite.max():.4f} s, "
        f"{int(unruptured.sum())} never ruptured"
    )


def main() -> int:
    root = Path(sys.argv[1])
    for problem in ("TPV101", "TPV102"):
        folder = root / problem
        if not folder.is_dir():
            fail(f"missing folder {folder}")
            continue
        present = {path.name for path in folder.iterdir() if path.is_file()}
        expected = {f"{name}.txt" for name in ON_FAULT_STATIONS}
        expected.add(f"{problem.lower()}_rupture_time.txt")
        if problem == "TPV102":
            expected |= {f"{name}.txt" for name in OFF_FAULT_STATIONS}
        for name in sorted(expected - present):
            fail(f"{problem}: missing {name}")
        for name in sorted(present - expected):
            fail(f"{problem}: unexpected file {name}")

        for name in sorted(ON_FAULT_STATIONS):
            path = folder / f"{name}.txt"
            if path.is_file():
                check_series(path, problem, ON_FAULT_FIELDS)
        if problem == "TPV102":
            for name in sorted(OFF_FAULT_STATIONS):
                path = folder / f"{name}.txt"
                if path.is_file():
                    check_series(path, problem, OFF_FAULT_FIELDS)
        contour = folder / f"{problem.lower()}_rupture_time.txt"
        if contour.is_file():
            check_contour(contour, problem)
        notes.append(f"{problem}: {len(present)} files")

    print("\n".join(notes))
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem_text in problems:
            print("  -", problem_text)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
