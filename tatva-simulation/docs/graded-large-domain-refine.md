# TPV101/TPV102 graded large-domain runs with a refined outer mesh

These runs repeat [graded-large-domain.md](graded-large-domain.md) with a finer,
more gradual outer mesh, and they additionally write the fault plane in the
UGUCA binary dump format so that one animation script can read Tatva and UGUCA
results alike. Domain size, the 100 m core, the physics, and the 15 s window are
unchanged, so the only difference against the previous run is how the far field
is discretised.

## What changed

| Setting | Graded large | Graded large + refine |
|---|---:|---:|
| Minimum cell size | 100 m | 100 m |
| Maximum cell size | 750 m | 500 m |
| Maximum adjacent-cell growth ratio | 1.05 | 1.04 |
| Slurm time limit | 08:00:00 | 16:00:00 |
| Operator batch size (TPV101 / TPV102) | 65536 / 32768 | 16384 / 16384 |
| Fault-plane dump | none | UGUCA binary, every 0.1 s |

Everything else — domain bounds, the 100 m fine box, element type, Lysmer
dashpots, duration, checkpoint interval — is identical to the previous run.

### Why 500 m and 1.04

The outer mesh only has to carry waves away from the fault without dispersing or
reflecting them before the 15 s window closes. With `Cs = 3464 m/s`, a linear
`Hexahedron8` mesh wants roughly ten cells per shear wavelength:

| Cell size | Cells per wavelength at 1 Hz | Frequency resolved at 10 cells/λ |
|---:|---:|---:|
| 100 m (core) | 34.6 | 3.46 Hz |
| 500 m (new outer) | 6.9 | 0.69 Hz |
| 750 m (old outer) | 4.6 | 0.46 Hz |

750 m cells only resolve about 0.46 Hz, which is inside the band the rupture
radiates, so part of the outgoing field was being dispersed by the mesh rather
than by the physics. Dropping the growth ratio as well lengthens the transition,
so the impedance contrast between neighbouring cells is smaller where the wave
leaves the fine region.

500 m is not where this started. 300 m with growth 1.03 was submitted first,
which is what the wall-clock budget allows, and both benchmarks ran the H200 out
of memory; see the measured ceiling below. GPU memory, not the 16 h limit, is
what caps these meshes.

## TPV101

| Quantity | Graded large | Refined |
|---|---:|---:|
| Domain x | -50 to 50 km | -50 to 50 km |
| Domain y | -35 to 50 km | -35 to 50 km |
| Positive-z half-domain | 0 to 30 km | 0 to 30 km |
| 100 m x interval | -18 to 18 km | -18 to 18 km |
| 100 m y interval | -3 to 18 km | -3 to 18 km |
| 100 m z interval | 0 to 8 km | 0 to 8 km |
| Cells (x, y, z) | 494, 344, 134 | 532, 382, 146 |
| Elements | 22,771,424 | 29,670,704 |
| Nodes | 23,054,625 | 30,008,433 |
| Degrees of freedom | 69,163,875 | 90,025,299 |
| Maximum cell size | 750 m | 500 m |
| Realised growth ratio | 1.0492 | 1.0382 |

The element count grows by 1.30x. The 750 m run spent 10,731 s inside the time
loop (15,000 steps at dt = 1.0e-3 s, including 87 s of first-step compilation)
out of 3:00:07 of job wall time. Element-count scaling puts the refined time loop
near 3.9 h, well inside the 16 h limit.

## TPV102

| Quantity | Graded large | Refined |
|---|---:|---:|
| Domain x | -50 to 50 km | -50 to 50 km |
| Domain y | 0 to 50 km | 0 to 50 km |
| Positive-z half-domain | 0 to 30 km | 0 to 30 km |
| 100 m x interval | -18 to 18 km | -18 to 18 km |
| 100 m y interval | 0 to 18 km | 0 to 18 km |
| 100 m z interval | 0 to 12 km | 0 to 12 km |
| Cells (x, y, z) | 494, 247, 168 | 532, 266, 178 |
| Elements | 20,499,024 | 25,189,136 |
| Nodes | 20,746,440 | 25,473,669 |
| Degrees of freedom | 62,239,320 | 76,421,007 |
| Maximum cell size | 750 m | 500 m |
| Realised growth ratio | 1.0499 | 1.0382 |

The element count grows by 1.23x. The 750 m run spent 9,826 s in the time loop
out of 2:45:21 of job wall time, so the refined run is expected near 3.4 h. The
physical free surface at y=0 is unchanged.

## GPU memory: what actually limits these meshes

Four jobs died with `CUDA_ERROR_OUT_OF_MEMORY` before this configuration ran, so
the numbers below are worth keeping. `nvidia-smi` says nothing useful here,
because JAX preallocates its pool either way; the useful numbers come from the
`device memory:` line `tpv101.py` prints after the first output, using the JAX
allocator's own `memory_stats()`.

| Configuration | Elements | Steady use | Peak | Result |
|---|---:|---:|---:|---|
| TPV101, 300 m / 1.03 | 46,342,504 | - | - | OOM before the first output (302407) |
| TPV102, 300 m / 1.03 | 35,569,560 | - | - | OOM after the first output (302409) |
| TPV101, 500 m / 1.04, pool 0.94 | 29,670,704 | 6.94 GiB | 97.82 GiB | OOM loading the CUBIN (302462) |
| TPV102, 500 m / 1.04, pool 0.94 | 25,189,136 | 5.89 GiB | 83.04 GiB | OOM loading the CUBIN (302463) |
| TPV101, 500 m / 1.04, pool 0.85 | 29,670,704 | 6.94 GiB | 97.82 GiB | runs |
| TPV102, 500 m / 1.04, pool 0.85 | 25,189,136 | 5.89 GiB | 83.04 GiB | runs |

Two things come out of that table.

**The peak is a compile-time transient and scales linearly with the element
count.** 97.82/83.04 = 1.178 against an element ratio of 29.67/25.19 = 1.178, so
budget about **3.3 GiB per million elements**. Steady-state use is only about
7 GiB. That also settles the first two failures: 46.3 million elements need
roughly 153 GiB of transient, which never had a chance on a 143 GB H200.

**The pool must not take the whole device.** The 0.94 runs failed with the pool
only 74% used, because the error is not a pool allocation at all — it is the
CUDA module image of the compiled executable, which lives outside the pool:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: [0] Failed to load in-memory
CUBIN (compiled for a different GPU?).: CUDA_ERROR_OUT_OF_MEMORY: out of memory
[executable_name='jit_advance_output']
```

JAX warns that this program captures gigabytes of constants during lowering
(2.74 GB at 22.8 million elements), and those constants are part of that image.
At `XLA_PYTHON_CLIENT_MEM_FRACTION=0.94` only 6% of the device, about 8.6 GB, is
left for it, and at 30 million elements that is no longer enough.
`XLA_MEMORY_FRACTION=0.85` gives a 118.8 GiB pool, comfortably above the
97.8 GiB peak, and leaves about 21 GB for the module image.

So the usable envelope on one H200 is roughly `elements x 3.3 GiB/M < 0.85 x
143 GB`, or about **34 million elements**, which is why 500 m / 1.04 is the
finest outer mesh in this configuration rather than the 300 m / 1.03 that the
wall-clock budget alone would have allowed.

The dump itself was also a contributor and has been changed: it first gathered
its window in a separate `jax.jit` function, which meant a second large program
resident beside `jit_advance_output`. The window is now sliced out of the fields
the step has already computed, so dumping compiles no code of its own and costs
a few hundred kilobytes per frame. That change moves the dumped
velocity-dependent fields from the half-step velocity onto the step's centred
velocity, which is the same evaluation the SCEC station output uses; on a smoke
run the difference is 1.4e-5 relative in `cohesion_0` and `top_disp_0` and
`theta` are unchanged.

The August 750 m runs used `tatva-v0.11.4-gpu`, which no longer exists on NANO4;
the runs since then use the `/work/gauss112/tatva` checkout, so the table mixes
two library builds.

## Fault-plane dump in UGUCA format

`uguca_dump.py` writes the same five-part layout UGUCA produces with
`Dumper::Format::Binary`:

```
<run name>.info     keys pointing at the other files and the data folder
<run name>.fields   "<field> <file>" per registered field
<run name>.time     "<time step> <physical time>" per frame
<run name>.coord    "x y z" per node, ASCII, %.10e
<run name>-DataFiles/<field>.out   little-endian float32, frame after frame
```

Registered fields mirror `benchmarks/TPV101/TPV101.cc`:

| Field | Meaning | Unit |
|---|---|---|
| `cohesion_0` | shear traction along strike | Pa |
| `top_disp_0` | top-side displacement along strike (half the slip) | m |
| `top_velo_0` | top-side velocity along strike (half the slip rate) | m/s |
| `theta` | rate-and-state state variable | s |

`top_disp_0` and `top_velo_0` are halved on purpose: UGUCA's
`UnimatShearInterface` defines the gap as twice the top-surface field, so a
reader that reconstructs slip as `2 * top_disp_0` gets the same answer for both
codes.

Two conventions differ from a UGUCA dump and matter when plotting:

- Nodes are written with the second in-plane axis varying fastest, as UGUCA does,
  and the Tatva down-dip axis `y` is written into the UGUCA `z` column. That axis
  therefore increases **downward**; UGUCA's `z` increases upward.
- Coordinates are the true fault coordinates, so `x` runs from -18 to 18 km
  rather than starting at zero.

Only the uniform 100 m window is dumped (`x` in [-18, 18] km, `y` in [-3, 18] km
for TPV101 and [0, 18] km for TPV102). The graded part of the fault plane is left
out because the UGUCA format assumes a constant node spacing. That gives
361 x 211 nodes for TPV101 (184 MB for 151 frames at 0.1 s) and 361 x 181 nodes
for TPV102 (158 MB).

Dumping is driven by the `UGUCA_DUMP_*` environment variables read by
`Tatva-TPV101-GPU-100m-15s.slurm` and `Tatva-TPV102-GPU-100m-15s.slurm`, which
map onto the `--uguca-dump-*` options of `tpv101.py` and `tpv102.py`. When
`UGUCA_DUMP_INTERVAL` is unset nothing is written, so the earlier scripts behave
exactly as before. A resumed run truncates the dump back to the checkpoint time
before appending, so restarts cannot duplicate frames.

## Reproduction

These runs live in their own checkout so the validated `tatva-simulation` tree
stays untouched. On NANO4:

```bash
cd /work/gauss112/SCEC-Code-Verification/SCEC-Code-Validation
git clone https://github.com/Gauss963/TPV101.git tatva-simulation-refine
cd tatva-simulation-refine
ln -s ../tatva-simulation/output output
ln -s ../tatva-simulation/reference reference
ln -s ../tatva-simulation/Plot Plot
sbatch Tatva-TPV101-GPU-100m-15s-graded-large-refine.slurm
sbatch Tatva-TPV102-GPU-100m-15s-graded-large-refine.slurm
```

`output`, `reference`, and `Plot` are symlinks back to the original tree, so the
results, the SCEC reference data, and the plots all stay in one place. The refine
scripts export `TATVA_SIM_ROOT`, which the shared
`Tatva-TPV10x-GPU-100m-15s.slurm` scripts honour; with that variable unset those
scripts still default to the original `tatva-simulation` directory.

Each job compares its output against UGUCA, SPECFEM3D, and the corresponding
750 m outer-mesh run, so the plots show the effect of the refinement directly.
Checkpoints are written every 0.5 simulation seconds and are resumed
automatically when the same run is resubmitted.
