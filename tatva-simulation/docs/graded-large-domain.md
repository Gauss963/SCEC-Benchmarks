# TPV101/TPV102 graded large-domain runs

These runs test whether the late horizontal-shear-stress mismatch is caused by
reflections from the finite Tatva domain. They retain the validated 100 m mesh
around the RSF fault and all requested stations, then grade smoothly toward a
larger outer boundary.

## Shared mesh rules

- Minimum cell size: 100 m
- Maximum cell size: 750 m
- Maximum adjacent-cell growth ratio: 1.05
- Element: conforming Tatva `Hexahedron8`; there are no hanging nodes
- Time step: based on the unchanged 100 m minimum cell and local contact
  mass/area ratio
- Outer boundary: axis-aligned Lysmer dashpots
- Duration: 15 s
- GPU allocation: one NVIDIA H200, with an 8 hour Slurm limit

The full RSF transition envelope remains in the 100 m region. TPV102 also
retains 100 m cells through z=12 km so all free-surface stations remain in the
fine mesh.

## TPV101

| Quantity | Value |
|---|---:|
| Domain x | -50 to 50 km |
| Domain y | -35 to 50 km |
| Positive-z half-domain | 0 to 30 km |
| 100 m x interval | -18 to 18 km |
| 100 m y interval | -3 to 18 km |
| 100 m z interval | 0 to 8 km |
| Cells (x, y, z) | 494, 344, 134 |
| Elements | 22,771,424 |
| Degrees of freedom | 69,163,875 |

The nearest outer boundary is 32 km from the RSF transition envelope in x and
y. A reflected shear wave therefore needs about 18.5 s for a round trip. The
z-boundary round trip is about 17.3 s. Both exceed the 15 s validation window.

The preceding uniform-expanded run used 22,080,000 elements and took 5:13:10.
Element-count scaling predicts about 5:23 for the simulation, leaving more than
two hours for compilation, validation, comparison, and plotting.

## TPV102

| Quantity | Value |
|---|---:|
| Domain x | -50 to 50 km |
| Domain y | 0 to 50 km |
| Positive-z half-domain | 0 to 30 km |
| 100 m x interval | -18 to 18 km |
| 100 m y interval | 0 to 18 km |
| 100 m z interval | 0 to 12 km |
| Cells (x, y, z) | 494, 247, 168 |
| Elements | 20,499,024 |
| Degrees of freedom | 62,239,320 |

The x and upper-y shear-wave round trips are about 18.5 s, and the z-boundary
round trip is about 17.3 s. The physical free surface at y=0 is unchanged.

The preceding uniform-expanded run used 23,040,000 elements and took 5:28:13.
Element-count scaling predicts about 4:52 for the new simulation.

## Reproduction

Submit the following scripts from the NANO4 project directory:

```bash
sbatch Tatva-TPV101-GPU-100m-15s-graded-large.slurm
sbatch Tatva-TPV102-GPU-100m-15s-graded-large.slurm
```

Each script compares its output against UGUCA, SPECFEM3D, and the corresponding
uniform-expanded Tatva result. Checkpoints are written every 0.5 simulation
seconds and are resumed automatically when the same run is resubmitted.
