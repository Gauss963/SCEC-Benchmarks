# TPV101 and TPV102 expanded-domain dimensions

This document records the exact geometry used by the 100 m, 15 s expanded-domain
Tatva runs submitted on 2026-08-14:

- TPV101: Slurm job `260896`, run name
  `tpv101_100m_15s_xy_expanded_gpu1_h200`
- TPV102: Slurm job `260897`, run name
  `tpv102_100m_15s_xy_expanded_gpu1_h200`

The local and NANO4 copies of `tpv101.py`, `tpv102.py`, the two base GPU Slurm
scripts, and the two expanded-domain Slurm scripts had identical SHA-256 hashes
when this document was prepared.

## Coordinate convention and symmetry

| Coordinate | Meaning |
| --- | --- |
| `x` | Along strike on the fault |
| `y` | Down-dip/depth coordinate on the fault |
| `z` | Fault-normal coordinate |

The fault is the plane `z = 0`. Both runs use fault-symmetry reduction and
explicitly mesh only the `z >= 0` block. Therefore, the listed computational
domain is the represented half-domain. The equivalent full physical domain is
obtained by reflecting it across `z = 0`.

TPV101 is a finite full-space problem in `y`: both `y` boundaries are absorbing.
TPV102 is a half-space problem: `y = 0` is a traction-free surface and is not an
absorbing boundary.

## Overall computational domains

All lengths in this section are kilometres.

| Quantity | TPV101 expanded | TPV102 expanded |
| --- | ---: | ---: |
| `x` extent | `[-30, 30]` | `[-30, 30]` |
| `y` extent | `[-16, 30]` | `[0, 32]` |
| Meshed `z` extent | `[0, 8]` | `[0, 12]` |
| Half-domain size | `60 x 46 x 8` | `60 x 32 x 12` |
| Equivalent full size | `60 x 46 x 16` | `60 x 32 x 24` |
| Meshed half-domain volume | `22,080 km^3` | `23,040 km^3` |
| Fault-plane size | `60 x 46` | `60 x 32` |
| Fault-plane area | `2,760 km^2` | `1,920 km^2` |

The expanded scripts change only the in-plane `x` and `y` extents. The
fault-normal `z` extents remain those of the validated `hpc-100m` presets.

### Difference from the original 100 m domains

| Quantity | TPV101 original | TPV101 expanded | TPV102 original | TPV102 expanded |
| --- | ---: | ---: | ---: | ---: |
| Domain size, represented half | `52 x 39 x 8 km` | `60 x 46 x 8 km` | `52 x 27 x 12 km` | `60 x 32 x 12 km` |
| `x` extent | `[-26, 26] km` | `[-30, 30] km` | `[-26, 26] km` | `[-30, 30] km` |
| `y` extent | `[-12, 27] km` | `[-16, 30] km` | `[0, 27] km` | `[0, 32] km` |
| Meshed half-domain volume | `16,224 km^3` | `22,080 km^3` | `16,848 km^3` | `23,040 km^3` |

## Mesh and problem size

Both cases use uniform `100 m` Hexahedron8 elements.

| Quantity | TPV101 expanded | TPV102 expanded |
| --- | ---: | ---: |
| Cells in `x` | 600 | 600 |
| Cells in `y` | 460 | 320 |
| Cells in represented `z` half | 80 | 120 |
| Elements | 22,080,000 | 23,040,000 |
| Nodes | 22,441,941 | 23,343,441 |
| Displacement DOFs | 67,325,823 | 70,030,323 |
| Fault nodes | 277,061 | 192,921 |
| Duration | 15 s | 15 s |
| Output interval | 0.005 s | 0.005 s |
| Internal time step | 0.000555556 s | 0.000555556 s |
| Outputs | 3,001 | 3,001 |

## RSF regions on the fault

Rate-and-state friction is evaluated on the entire meshed fault plane. The
finite rectangle below is specifically the velocity-weakening core, not the
only part of the fault where RSF is active.

The spatial profile is centred at `(x, y) = (0, 7.5 km)`.

| Region/property | TPV101 | TPV102 |
| --- | ---: | ---: |
| Velocity-weakening core | `x = [-15, 15] km`, `y = [0, 15] km` | Same |
| Core size | `30 x 15 km` | `30 x 15 km` |
| Core area | `450 km^2` | `450 km^2` |
| Transition width | 3 km on each available side | 3 km on each available side |
| Non-fully-strengthening envelope | `x = (-18, 18) km`, `y = (-3, 18) km` | `x = (-18, 18) km`, `y = [0, 18) km` inside the half-space |
| Fully velocity-strengthening region | Outside the transition envelope | Outside the in-domain transition envelope |

For TPV102, the nominal lower `y` transition from `-3` to `0 km` lies outside
the half-space. The velocity-weakening core therefore reaches the free-surface
edge at `y = 0`; only the upper `y = 15` to `18 km` transition is represented.

### Friction values

| Property | Value |
| --- | ---: |
| Core direct effect `a` | 0.008 |
| State effect `b` | 0.012 |
| Core `a - b` | -0.004, velocity weakening |
| Far-field direct effect `a` | 0.016 |
| Far-field `a - b` | +0.004, velocity strengthening |
| Characteristic slip `D_c` | 0.02 m |
| Reference friction | 0.6 |
| Reference velocity | `1e-6 m/s` |
| Initial normal stress | 120 MPa |
| Initial shear stress | 75 MPa |

The 3 km transition is a smooth boxcar profile. At transition corners the
one-dimensional `x` and `y` weights are multiplied, so the profile is smooth
rather than a piecewise rectangular step.

### Nucleation region

Both cases use the same circular nucleation perturbation:

- Centre: `(x, y) = (0, 7.5 km)`
- Radius: `3 km` (diameter `6 km`)
- Maximum shear-stress perturbation: `25 MPa`
- Temporal ramp time: `1 s`

## Absorbing boundaries and dimensions

There is no finite-thickness PML or sponge volume in these runs. Absorption is
implemented by zero-thickness, axis-aligned Lysmer dashpots on the outer mesh
surfaces. Consequently, an "absorbing-region thickness" is not defined.

For each absorbing face, the normal velocity component uses `rho * c_p * A`
and the two tangential components use `rho * c_s * A`. Contributions are added
at edges and corners. The fault plane `z = 0` never receives an absorbing
dashpot.

### TPV101 absorbing faces

| Face | Surface extent | Face size |
| --- | --- | ---: |
| `x = -30 km` | `y = [-16, 30]`, `z = [0, 8] km` | `46 x 8 km` |
| `x = +30 km` | `y = [-16, 30]`, `z = [0, 8] km` | `46 x 8 km` |
| `y = -16 km` | `x = [-30, 30]`, `z = [0, 8] km` | `60 x 8 km` |
| `y = +30 km` | `x = [-30, 30]`, `z = [0, 8] km` | `60 x 8 km` |
| `z = +8 km` | `x = [-30, 30]`, `y = [-16, 30] km` | `60 x 46 km` |

Distances from the outer edge of the RSF transition envelope to an absorbing
boundary are `12 km` in both `x` directions, `13 km` toward `y = -16 km`,
`12 km` toward `y = +30 km`, and `8 km` in the fault-normal direction.

### TPV102 absorbing and free surfaces

| Face | Type | Surface extent | Face size |
| --- | --- | --- | ---: |
| `x = -30 km` | Lysmer absorbing | `y = [0, 32]`, `z = [0, 12] km` | `32 x 12 km` |
| `x = +30 km` | Lysmer absorbing | `y = [0, 32]`, `z = [0, 12] km` | `32 x 12 km` |
| `y = +32 km` | Lysmer absorbing | `x = [-30, 30]`, `z = [0, 12] km` | `60 x 12 km` |
| `z = +12 km` | Lysmer absorbing | `x = [-30, 30]`, `y = [0, 32] km` | `60 x 32 km` |
| `y = 0 km` | Traction-free, not absorbing | `x = [-30, 30]`, `z = [0, 12] km` | `60 x 12 km` |

Distances from the outer edge of the represented RSF transition envelope to an
absorbing boundary are `12 km` in both `x` directions, `14 km` toward
`y = +32 km`, and `12 km` in the fault-normal direction. There is no lower `y`
absorbing buffer because `y = 0` is the physical free surface.

## Source files

The dimensions above are defined by:

- `tpv101.py`: base geometry, RSF profile, mesh, and Lysmer dashpots
- `tpv102.py`: TPV102 free-surface and fault-normal extent overrides
- `Tatva-TPV101-GPU-100m-15s-expanded.slurm`: TPV101 expanded `x/y` overrides
- `Tatva-TPV102-GPU-100m-15s-expanded.slurm`: TPV102 expanded `x/y` overrides
- `Tatva-TPV101-GPU-100m-15s.slurm` and
  `Tatva-TPV102-GPU-100m-15s.slurm`: common 100 m, 15 s launch settings
