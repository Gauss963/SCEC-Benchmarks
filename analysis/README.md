# Analysis

Rendering scripts for the TPV101/TPV102 results. Both write a sequence of PNG
frames and encode them into an H.264 MP4, so they need `numpy`, `matplotlib`,
and `ffmpeg` on `PATH`.

## `TPVShearAnimation.py`

Animates the horizontal shear stress on the rupture plane from a UGUCA binary
fault dump. The same layout is produced by UGUCA's own `benchmarks/TPV10x` and
by `tpv10x.py --uguca-dump-interval ...`, so one script covers both codes and
the two can be compared frame by frame.

Everything is drawn in the SCEC fault convention: distance along strike `j`
horizontally, distance down dip `k` increasing downward, `k = 0` at the top of
the 30 x 15 km rupture patch (the free surface for TPV102), and the hypocenter
at `j = 0`, `k = 7.5 km`. Each dump's own coordinates are mapped into that frame
by `--convention`, which defaults to `auto`:

| convention | mapping |
|---|---|
| `tatva` | already SCEC: `j = x`, `k = z` |
| `uguca-fullspace` | `j = x - Lx/2`, `k = z - (Lz/2 - 7.5 km)` |
| `uguca-freesurface` | `j = x - Lx/2`, `k = Lz/2 - z`; the benchmark's mirror image is dropped |

The figure shows the shear-stress field, the velocity-weakening patch, the
hypocenter, the nine SCEC on-fault stations, and a time-series panel with a
cursor. Station curves are read from `faultst*.txt` when those sit beside the
dump, and sampled from the dumped field otherwise.

```bash
# Tatva
python analysis/TPVShearAnimation.py \
    <output>/tpv101_100m_15s_graded_large_refine_gpu1_h200/tpv101_100m_15s_graded_large_refine_gpu1_h200 \
    --output-dir analysis/out/TPV101_Tatva --vmin 30 --vmax 135 --overwrite

# UGUCA, same window and color scale
python analysis/TPVShearAnimation.py \
    <uguca dump>/TPV101_Nx1440_Nz720_s2.00_tf0.35_npc1 \
    --output-dir analysis/out/TPV101_UGUCA --vmin 30 --vmax 135 --overwrite
```

### Stills for a document

`--snapshot TIME` writes the single frame nearest that time instead of a video.
A `.pdf` target keeps the axes, labels and colourbar as vector text and
rasterises only the field itself at `--dpi`, and the surrounding white space is
trimmed, so the file drops straight into a figure environment:

```bash
python analysis/TPVShearAnimation.py <dump base> \
    --snapshot 6.0 --snapshot-path figures/tpv101_uguca_t6.pdf \
    --vmin 30 --vmax 135 --width 1920 --height 1440 --dpi 300 --overwrite
```

`--width`, `--height` and `--dpi` set the physical size rather than a pixel
count: 1920 x 1440 at 300 dpi is a 6.4 x 4.8 inch figure, and the type is sized
from that. A still keeps the fault plane only, because two of them printed side
by side at half the text width leave the station legend too small to read; pass
`--snapshot-panels full` to keep the time series as well.

Give both runs the same `--vmin/--vmax` and stack them for a side-by-side
comparison:

```bash
ffmpeg -i TPV101_UGUCA.mp4 -i TPV101_Tatva.mp4 \
  -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p TPV101_UGUCA_vs_Tatva.mp4
```

## `FEMRuptureAnimation.py`

For Tatva runs that carry no fault dump — every run before
`--uguca-dump-interval` existed — the shear-stress field simply was not saved.
This script animates what those runs do contain: the ruptured area growing out
of `<problem>_internal_diagnostics.npz` with the current front outlined, and the
nine on-fault stations coloured by their instantaneous horizontal shear stress,
over the same time-series panel.

```bash
python analysis/FEMRuptureAnimation.py \
    <output>/tpv101_100m_15s_xy_expanded_gpu1_h200 \
    --output-dir analysis/out/TPV101_expanded --overwrite
```

Prefer `TPVShearAnimation.py` when a dump exists; it shows the field itself
rather than the arrival-time reconstruction.
