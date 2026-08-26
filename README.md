# SCEC-Benchmarks

Work on the [SCEC/USGS Spontaneous Rupture Code Verification
Project](http://scecdata.usc.edu/cvws/), currently TPV101 and TPV102: spontaneous
rupture on a vertical strike-slip fault with regularized rate-and-state friction
and the Dieterich ageing law, in a whole space (TPV101) and a half space with a
free surface (TPV102).

The benchmarks are run with **Tatva**, a JAX finite-element code, and compared
against the public CVWS submissions — UGUCA (Chun-Yu Ke, spectral boundary
integral) and SPECFEM3D (Kaneko) — as well as against Tatva's own mesh and
domain variants.

This repository supersedes the earlier `TPV101` and `TPV102` repositories, which
split the same code in two and are no longer maintained.

## Layout

| Path | Contents |
|---|---|
| `tatva-simulation/` | Drivers, Slurm scripts, validation tooling, and benchmark notes |
| `tatva-simulation/docs/` | Validation status, domain studies, and the refined-mesh runs |
| `analysis/` | Rendering scripts for the rupture animations |
| `tpv101_matlab/` | MATLAB helpers for the TPV101 initial conditions (`boxcar.m`, `input_DR1.m`, `load_DR1.m`) |
| `tmp/tatva-simulation-prototypes/` | An abandoned explicit-integration prototype, kept for reference |
| `tmp/pdfs/` | Text and page renders extracted from the SCEC benchmark documents |
| `SCEC_validation_ageing_law.pdf` | SCEC's TPV101 problem description |
| `uploadTPV101.pdf` | SCEC's TPV101/TPV102 data-format instructions |

Start with [tatva-simulation/README.md](tatva-simulation/README.md) for the
implemented specification and how to run a case, and
[tatva-simulation/docs/validation.md](tatva-simulation/docs/validation.md) for
what has and has not been verified.

## The runs

| Configuration | Domain and mesh | Notes |
|---|---|---|
| `Tatva-TPV10x-GPU-100m-15s.slurm` | The shared driver script; every other script sets environment variables and execs it | |
| `...-expanded.slurm` | Uniform 100 m over an enlarged domain | |
| `...-graded-large.slurm` | Large domain, 100 m core graded to 750 m | [docs/graded-large-domain.md](tatva-simulation/docs/graded-large-domain.md) |
| `...-graded-large-refine.slurm` | Same domain graded to 500 m, plus a UGUCA-format fault dump | [docs/graded-large-domain-refine.md](tatva-simulation/docs/graded-large-domain-refine.md) |

The refined runs also write the fault plane in the UGUCA binary dump format
(`uguca_dump.py`), so a Tatva result and a UGUCA result can be read by the same
analysis script and animated side by side — see
[analysis/README.md](analysis/README.md).

## What is not in the repository

Simulation output, figures, and reference data are excluded by `.gitignore`
because they run to several GB:

- `tatva-simulation/output/` and `tatva-simulation/Plot/` — regenerate by
  submitting the Slurm scripts;
- `tatva-simulation/reference/` — download with `fetch_scec_reference.py` and
  `fetch_scec_site_plots.py`.

Two published journal articles that were kept alongside this work are also
excluded rather than redistributed here:

- Harris, Barall & Archuleta (2009), *Seismological Research Letters*;
- Erickson et al. (2020), *Seismological Research Letters*, on the SEAS
  verification exercise.

## Requirements

Tatva itself lives in a separate repository and is not vendored here. The Slurm
scripts resolve it from the installed package, so any checkout that
`import tatva` can find will do; set `TATVA_ROOT` to override. The drivers need
JAX with CUDA for the production meshes, and the 100 m cases are sized for a
single H200.
