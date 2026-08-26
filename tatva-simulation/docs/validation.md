# TPV101 Validation Status

Validation date: 2026-08-11

Long-duration follow-up started on 2026-08-08. The symmetry-reduced 100 m
first-step preflight was terminated after severe compression and approximately
15.4 GiB of swap use on the 24 GB workstation. A 149.425 m first-output test
completed, but sustained stepping continued to swap at roughly 170--200 MB/min.
The minimally coarser 158.537 m case completed the full 15 s simulation on
2026-08-09 in 134,019 s (37.23 h). Its nine station files, complete rupture
contour, and SCEC ASCII format pass validation.

## Scope

This workflow verifies the SCEC TPV101 regularized rate-and-state friction law,
the Dieterich ageing-state update, Tatva contact integration, and SCEC-compatible
output. Dynamic convergence is assessed against six public submissions downloaded
from the SCEC/USGS comparison server: FaultMod, DFM, MDSBI, SPECFEM3D, UGUCA,
and BI. Their native fault spacing is 50--100 m. The primary implementation
reference is CVWS `User: ke`, listed by SCEC as `Chun-Yu Ke - Spectral Boundary
Integral - uguca`; the other five submissions are secondary cross-checks.

## Results

| Check | Result |
| --- | --- |
| Exact RSF strength and Equation (6) inverse | Pass |
| Initial `theta` in the VW region | `1.606238999213454e9 s` |
| Recovered initial traction | `75 MPa` at `V=1e-12 m/s`, `sigma=120 MPa` |
| Ageing-law positivity and JIT compatibility | Pass |
| 0.1 s smoke dynamics | Pass; no spurious rupture |
| Nine station files and rupture contour schema | Pass |
| Complete 1 km dump from 0 to 12 s | Pass format; not dynamically converged |
| Six public raw submissions | Pass; nine stations plus rupture contour per code |
| Official CVWS-generated public plots | Pass; 438 GIF files (73 per code) |
| One-half symmetry reduction | Pass; 1 km primary fields and rupture arrivals equal full domain |
| Complete 158.537 m dump from 0 to 15 s | Pass; 3,001 samples at all nine stations |
| 158.537 m fault rupture | Pass; all 17,955 VW nodes ruptured by 8.065 s |

The SCEC upload convention defines normal stress as positive in compression.
Tatva follows this convention. Three historical public submissions store normal
compression with the opposite sign, so the comparison/plotting layer normalizes
only that column before plotting or computing station RMS. Downloaded source files
and official CVWS GIF plots are preserved byte-for-byte.

The local nucleation and early rupture show clear convergence toward the primary
UGUCA 100 m reference:

| Mesh | Duration | Center arrival | Difference from UGUCA | Contour RMS vs UGUCA | Tatva/UGUCA ruptured nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 km | 2 s | 0.594643 s | +26.021 ms | 189.871 ms | 12 / 16 |
| 500 m | 2 s | 0.570297 s | +1.675 ms | 59.607 ms | 51 / 75 |
| 200 m | 2 s | 0.570256 s | +1.634 ms | 24.227 ms | 438 / 472 |

At the hypocentral center station, the 200 m normalized RMS differences from
UGUCA over 0--2 s are 1.73% for horizontal slip, 6.30% for slip rate, 1.76% for
horizontal shear stress, and 1.68% for log-theta. Across all six public
submissions, the median early-contour RMS decreases from 189.3 ms at 1 km, to
59.5 ms at 500 m, and 25.1 ms at 200 m. The 200 m run completed in 18,798 s
(5.22 h) with 6.82 GB peak resident memory and no swap reported by `/usr/bin/time`.

This is a strong early-time result, not a completed TPV101 validation. The 2 s
run reaches only the hypocentral one of the nine required stations; UGUCA reaches
the remaining stations at approximately 3.14--6.33 s. RSF must not be promoted
to the PMMA model on this evidence alone. The validation gate is a long-duration
fine-mesh run that reproduces all nine UGUCA station histories and the complete
rupture-time contour, followed by the recommended 100 m mesh check.

The 1 km long-duration solution propagates much too slowly and must not be used
as a reference-quality TPV101 result. Its purpose is to prove that the entire
Tatva-to-SCEC workflow runs and produces valid files. The 500 m checkpoint is a
positive convergence result, but it still under-resolves the approximately
200--270 m process-zone widths reported by the SCEC reference submissions. The
200 m result resolves that scale with only about one element: its convergence is
substantially better, but it is not a substitute for the recommended 100 m run.

### Full-duration 158.537 m comparison

The completed intermediate run follows both 100 m public references closely but
is systematically slower. Against UGUCA, its fault-wide rupture-time RMS, bias,
and maximum absolute difference are 137.6, +119.5, and 294.7 ms. Against
SPECFEM3D they are 175.5, +156.0, and 374.6 ms. The positive bias means that the
Tatva rupture front arrives later. All nodes rupture in both Tatva and each
reference over their common coordinate support.

Across the nine stations, the median Tatva arrival delay is 35.1 ms relative to
UGUCA and 40.4 ms relative to SPECFEM3D. Median normalized RMS waveform
differences for horizontal slip, slip rate, shear stress, and log-theta are
4.30%, 5.83%, 5.44%, and 7.28% against UGUCA, and 4.34%, 7.14%, 6.97%, and
7.58% against SPECFEM3D. The larger errors occur at the outer stations, while
the hypocentral station agrees closely. The comparison figures and complete
CSV/JSON statistics are in
`Plot/tpv101_validation_uguca_specfem3d_160m_15s/`.

### Invalidated 100 m MPI run

The first complete 100 m / 15 s MPI run (Slurm job 1015602) is invalid. The MPI
symmetry path doubled the fault-normal interface displacement along with the two
tangential components. The serial symmetry path correctly prescribes zero normal
gap. The MPI error produced up to 22.6778 MPa of artificial normal-stress
variation at the outer stations, changed the RSF strength, and made off-axis
rupture arrivals systematically early.

The near-hypocenter arrival remained close to UGUCA, but the median Tatva arrival
became 0.51--0.65 s early at radii of 10--14 km. This is a formulation error, not
evidence that a 100 m mesh converged to a faster rupture speed. The result and
its checkpoints are retained under `output/archive/invalid/` for diagnosis only.

The MPI implementation now doubles only the two tangential components and fixes
the normal jump to zero. The validator independently requires all station normal
stress histories to remain at the prescribed 120 MPa. A 3 s, 500 m serial/MPI
dynamic parity job crosses nucleation and the central MPI partition boundary and
is the regression gate before repeating the formal 100 m run. Operator batch
sizes 4,096 and 32,768 produced byte-identical 200 m
benchmark contours, so batching is not a numerical reduction behind the timing
difference.

## Required Next Level

The SCEC documentation recommends 100 m element size. After exact fault-plane
symmetry reduction, the `hpc-100m` preset still has approximately 49.5 million
displacement degrees of freedom and 16.2 million Hex8 elements. Its local
preflight exhausted practical memory, so the complete 100 m check remains an HPC
requirement. The 158.537 m run is an intermediate full-duration validation, not
a substitute for the recommended 100 m run. A publishable validation claim also
requires checks on finite-domain/Lysmer-boundary sensitivity. The six-code
comparison and a UGUCA-only primary comparison are automated by
`plot_tpv101_validation.py`.

Tatva v0.11.4 provides the `ExchangePlan` communication primitive used by this
workflow's along-strike MPI decomposition. Tatva currently has no native PML in
this checkout, so the implementation approximates the whole-space using a finite
3D domain with axis-aligned Lysmer dashpots. That approximation is separate from
the RSF constitutive verification and must remain explicit in any report.

## Reproduce

```bash
cd /Volumes/Gauss-T7/SCEC-Code-Validation/tatva-simulation
conda run -n tatva pytest -q test_tpv101.py
conda run -n tatva pytest -q /Volumes/Gauss-T7/tatva/tests/test_friction.py
conda run -n tatva python validate_scec_dump.py output/tpv101_coarse_1km
conda run -n tatva python compare_scec_reference.py \
  output/tpv101_200m_2s reference/ke_100m \
  --output output/tpv101_200m_2s/uguca_reference_comparison.json
conda run -n tatva python fetch_scec_site_plots.py --user all
conda run -n tatva python plot_tpv101_validation.py \
  --reference-user ke \
  --output-dir Plot/tpv101_validation_uguca \
  --candidate 'Tatva 1 km=output/tpv101_coarse_1km_2s' \
  --candidate 'Tatva 500 m=output/tpv101_500m_2s' \
  --candidate 'Tatva 200 m=output/tpv101_200m_2s'
```
