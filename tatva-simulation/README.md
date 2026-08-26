# Tatva SCEC TPV101 Rate-and-State Validation

SCEC TPV101 implemented with Tatva's 3D `Hexahedron8` finite-element operators:
a whole-space problem with regularized rate-and-state friction and the
Dieterich ageing law. It is an independent verification case from the linear
slip-weakening work, and it shares its driver with
[TPV102](https://github.com/Gauss963/TPV102), the half-space variant.

Validation status and the quantitative convergence limits are collected in
[docs/validation.md](docs/validation.md).

## Repository layout

| Path | Purpose |
|---|---|
| `tpv101.py` | TPV101 driver: mesh, friction, time loop, SCEC dump |
| `tpv102.py` | TPV102 driver; reuses `run_simulation` from `tpv101.py` |
| `uguca_dump.py` | Writes the fault plane in the UGUCA binary dump format |
| `validate_scec_dump.py` | Checks a dump against the SCEC format and initial conditions |
| `compare_scec_reference.py` | Station and rupture-time comparison against a reference submission |
| `plot_tpv101_validation.py` | Station, contour, and convergence figures |
| `analyze_horizontal_shear_tail.py` | Late horizontal-shear-stress diagnostics |
| `fetch_scec_reference.py`, `fetch_scec_site_plots.py` | Download public CVWS submissions and their site plots |
| `summarize_gpu_metrics.py` | Reduce an `nvidia-smi` sampling log to a summary |
| `test_tpv101.py`, `test_tpv102.py` | Unit and smoke-level regression tests |
| `Tatva-TPV10*.slurm` | NANO4 batch scripts |
| `docs/` | Benchmark notes for the large-domain and refined-mesh runs |

Both drivers are kept in both repositories because `tpv102.py` imports
`tpv101.py`, and because the refined-mesh Slurm scripts run either benchmark
from a single checkout.

## Implemented specification

- SCEC regularized rate-and-state friction:
  `tau = a sigma asinh[V/(2 V0) exp((f0 + b ln(V0 theta/L))/a)]`.
- Ageing law `theta_dot = 1 - V theta/L`, updated analytically at fixed slip
  rate over each step so that `theta > 0` is preserved.
- The spatially varying initial `theta` is inverted from TPV101 equation (6),
  so the whole fault sits exactly at `tau = 75 MPa` and `sigma = 120 MPa` for
  `V = 1e-12 m/s`.
- `a(x, y)` with its 3 km smooth transition, the 25 MPa / 3 km / 1 s smooth
  nucleation load, and the `V > 1 mm/s` rupture-arrival definition.
- The two fault sides are separate 3D Hex8 domains. Contact friction solves the
  velocity and the rate-and-state strength consistently in a local implicit
  step, which avoids the explicit blow-up that the low-velocity direct effect
  would otherwise cause.
- The computation uses an increment formulation around the initial stress.
  The outer boundary of the finite domain approximates the whole space with
  axis-aligned Lysmer dashpots.

## Running

Quick equation, format, and short-duration dynamics checks:

```bash
conda run -n tatva pytest -q test_tpv101.py
conda run -n tatva python tpv101.py --preset smoke --output-dir output/tpv101_smoke
conda run -n tatva python validate_scec_dump.py output/tpv101_smoke --allow-partial
```

A 12 s, 1 km coarse comparison dump:

```bash
conda run -n tatva python tpv101.py --preset coarse --output-dir output/tpv101_coarse_1km
conda run -n tatva python validate_scec_dump.py output/tpv101_coarse_1km
```

A 200 m case over the same domain and duration as the 1 km and 500 m
checkpoints:

```bash
conda run --no-capture-output -n tatva python tpv101.py \
  --preset hpc-200m --duration 2 --output-dir output/tpv101_200m_2s
```

`hpc-200m` uses the exact TPV101 mirror symmetry about `z = 0` and allocates
only one side of the Tatva Hex8 domain; the tangential slip and slip rate on
the fault are twice that side's values. The 1 km regression shows the main
station fields and the rupture arrival to be value-for-value identical to the
full two-sided model. The reduction brings the 200 m case down to 2,028,000
elements and 6,292,188 displacement degrees of freedom without changing the
constitutive law or the Tatva operator.

Long fine-mesh runs write atomic checkpoints; after an interruption, repeat the
same command with `--resume`:

```bash
conda run --no-capture-output -n tatva python tpv101.py \
  --preset hpc-160m --duration 15 \
  --checkpoint-path output/tpv101_160m_15s_checkpoint.npz \
  --checkpoint-interval 0.5 \
  --output-dir output/tpv101_160m_15s
```

Download the public UGUCA submission from the SCEC code-verification site along
with the site-generated GIF plots, then compare station by station and by
rupture time. CVWS `User: ke` is officially
`Chun-Yu Ke - Spectral Boundary Integral - uguca`:

```bash
conda run -n tatva python fetch_scec_reference.py --user ke
conda run -n tatva python fetch_scec_site_plots.py --user all
conda run -n tatva python compare_scec_reference.py \
  output/tpv101_200m_2s reference/ke_100m \
  --output output/tpv101_200m_2s/uguca_reference_comparison.json
```

Other public submissions listed by `fetch_scec_reference.py --help` download the
same way. Comparison reports are written into the candidate directory as
`reference_comparison.json`.

Replot all eight station variables, the rupture contours, and mesh convergence
for the Tatva 1 km, 500 m, and 200 m results against UGUCA:

```bash
conda run -n tatva python plot_tpv101_validation.py \
  --reference-user ke \
  --output-dir Plot/tpv101_validation_uguca \
  --candidate 'Tatva 1 km=output/tpv101_coarse_1km_2s' \
  --candidate 'Tatva 500 m=output/tpv101_500m_2s' \
  --candidate 'Tatva 200 m=output/tpv101_200m_2s'
```

Figures land in `Plot/tpv101_validation_uguca/` as identical PNG and PDF pairs,
with the per-station and contour numbers under its `stats/`. Dropping
`--reference-user ke` replots the six public solutions instead. The unmodified
CVWS GIFs stay in `reference/<user>_100m/site_plots/`.

Inspect the size of an HPC preset without allocating the mesh:

```bash
conda run -n tatva python tpv101.py --preset hpc-500m --dry-run
conda run -n tatva python tpv101.py --preset hpc-200m --dry-run
conda run -n tatva python tpv101.py --preset hpc-100m --dry-run
```

`--mesh-size`, `--z-extent`, `--duration`, `--output-dt`, and
`--operator-batch-size` override any preset.

## SCEC dump

Every output directory contains:

- nine `faultst*.txt` files with
  `t h-slip h-slip-rate h-shear-stress v-slip v-slip-rate v-shear-stress
  n-stress log-theta`, in SCEC units and sign conventions;
- `tpv101_rupture_time.txt` with columns `j k t`, where nodes that never
  ruptured carry `1.0E+09`;
- `validation_report.json`, the independent format and initial-condition check;
- `summary.json` with the mesh, time step, boundary approximation, and paths;
- `tpv101_internal_diagnostics.npz` for local diagnostics, not part of a SCEC
  submission.

## UGUCA-format fault dump

`uguca_dump.py` lets either driver additionally write the fault plane in the
UGUCA binary dump layout — `.info`, `.fields`, `.time`, `.coord`, and
`-DataFiles/<field>.out` — with the same field names as
`benchmarks/TPV101/TPV101.cc`: `cohesion_0`, `top_disp_0`, `top_velo_0`, and
`theta`. UGUCA and Tatva results can then be animated with one script.

Only the uniform part of the fault plane is dumped, because the format assumes a
constant node spacing; the graded region is left out. The Tatva down-dip axis
`y` is written into the UGUCA `z` column, so that axis increases downward.

```bash
python tpv101.py --preset hpc-100m --duration 15 \
  --graded-mesh --fine-x-min -18000 --fine-x-max 18000 \
  --fine-y-min -3000 --fine-y-max 18000 --fine-z-extent 8000 \
  --max-mesh-size 300 --mesh-growth-ratio 1.03 \
  --uguca-dump-interval 0.1 --output-dir output/<run name>
```

In the Slurm scripts the same thing is driven by `UGUCA_DUMP_INTERVAL`,
`UGUCA_DUMP_X_MIN`, and friends. With `UGUCA_DUMP_INTERVAL` unset nothing is
written, so the earlier scripts behave exactly as they did before.

## NANO4 batch runs

| Script | Domain and mesh |
|---|---|
| `Tatva-TPV10x-GPU-100m-15s.slurm` | Shared driver script; the others set environment variables and exec it |
| `Tatva-TPV10x-GPU-100m-15s-expanded.slurm` | Uniform 100 m over an enlarged domain |
| `Tatva-TPV10x-GPU-100m-15s-graded-large.slurm` | Large domain, 100 m core graded to 750 m |
| `Tatva-TPV10x-GPU-100m-15s-graded-large-refine.slurm` | Same domain graded to 300 m, plus the UGUCA dump |

`TATVA_SIM_ROOT` selects the checkout a job runs from, and `TATVA_ROOT` is
resolved from the installed `tatva` package unless it is set explicitly.

- [docs/expanded-domain.md](docs/expanded-domain.md)
- [docs/graded-large-domain.md](docs/graded-large-domain.md) — 750 m outer cells, growth ratio 1.05
- [docs/graded-large-domain-refine.md](docs/graded-large-domain-refine.md) — 300 m outer cells, growth ratio 1.03, with the UGUCA dump

## Validation levels and limits

`smoke` only exercises the equations, the stability of the integration, and the
output format; it cannot resolve the 3 km nucleation patch. `coarse` is enough
to inspect the full 12 s rupture behaviour, but a 1 km mesh is still not a
converged solution. Before any quantitative comparison against another code, at
least the 1 km, 500 m, and finer meshes should be compared, and `z_extent`
should be increased to check the influence of the artificial boundary.

Even after the symmetry reduction the 100 m preset carries about 49.5 million
degrees of freedom and 16.22 million Hex8 elements; a first output test on a
24 GB workstation produced roughly 15.4 GiB of swap, so that preset belongs on
an HPC node. The practical 158.537 m preset has about 12.43 million degrees of
freedom and 4.03 million elements, and it passed a five-output, zero-swap local
stress test, which makes it the intermediate case for a full 15 s validation.
Tatva has no native PML or MPI domain decomposition yet, so the whole space is
currently approximated by a finite domain with Lysmer dashpots. That
approximation has to be reported together with the mesh-convergence results;
a coarser dump must not be presented as a SCEC reference-quality solution.
