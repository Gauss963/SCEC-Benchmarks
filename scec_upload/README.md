# SCEC upload set

The files in this folder are the ones the SCEC code verification server expects
for TPV101 and TPV102, following `uploadTPV101.pdf` (February 10, 2007). Upload
them at <http://scecdata.usc.edu/cvws/cgi-bin/cvws.cgi> under **Upload Files**,
selecting each slot and its matching file.

They are copies of the refined graded-large-domain runs, which are the most
converged configuration in this repository:

| Benchmark | Source run |
|---|---|
| TPV101 | `tatva-simulation/output/tpv101_100m_15s_graded_large_refine_gpu1_h200` |
| TPV102 | `tatva-simulation/output/tpv102_100m_15s_graded_large_refine_gpu1_h200` |

Both passed `validate_scec_dump.py` on the source runs, and the assembled set
here passes `tatva-simulation/check_upload_set.py`, which checks this folder
against the specification directly:

```bash
python tatva-simulation/check_upload_set.py scec_upload
```

It verifies that each benchmark has exactly the expected files and no others,
that every header carries the required keys with the right `problem` and a real
author, that the field-list line and the per-column comment lines match the
specification, that `num_time_steps` agrees with the row count, that the time
column is strictly increasing with a uniform step equal to the declared
`time_step`, that no value is non-finite, that each station's `location` line
agrees with its file name, that the on-fault series start from 75 MPa shear,
120 MPa normal and zero slip, and that the contour files hold three columns of
unique nodes inside the allowed `j` and `k` ranges with `1.0E+09` for anything
that never ruptured.

Against the UGUCA reference (CVWS `User: ke`, 100 m) the normalized RMS
differences over the nine on-fault stations are:

| Quantity | TPV101 mean / max | TPV102 mean / max |
|---|---:|---:|
| Horizontal slip | 0.52% / 0.72% | 1.36% / 1.69% |
| Horizontal slip rate | 3.63% / 4.73% | 4.67% / 6.07% |
| Horizontal shear stress | 1.02% / 1.27% | 1.58% / 3.29% |
| log10 theta | 0.76% / 0.93% | 1.33% / 2.90% |

## What goes where

### TPV101 — 10 files

| File | Server slot |
|---|---|
| `faultst-120dp030.txt` … `faultst120dp120.txt` (9 files) | On-fault time series, one per station |
| `tpv101_rupture_time.txt` | Contour plot file |

TPV101 is a whole-space problem and has no off-fault stations.

### TPV102 — 16 files

| File | Server slot |
|---|---|
| `faultst-120dp030.txt` … `faultst120dp120.txt` (9 files) | On-fault time series, one per station |
| `body-060st-120dp000.txt` … `body060st120dp000.txt` (6 files) | Off-fault time series, one per station |
| `tpv102_rupture_time.txt` | Contour plot file |

## Format, as delivered

On-fault files carry 3001 rows at a uniform 0.005 s step over 0 to 15 s, with the
nine fields `t h-slip h-slip-rate h-shear-stress v-slip v-slip-rate
v-shear-stress n-stress log-theta`. Off-fault files carry the seven fields
`t h-disp h-vel v-disp v-vel n-disp n-vel`. The contour files carry `j k t` with
`1.0E+09` for nodes that never ruptured, at 100 m node spacing over
`j` in [-14900, 14900] m and `k` in [100, 15000] m.

Each file has the header the specification asks for: `problem`, `author`, `date`,
`code`, `code_version`, `element_size`, `time_step`, `num_time_steps`,
`location`, and one comment line per data column, followed by the field-list line
and the data.

## One thing to check before uploading

**The off-fault file names.** `uploadTPV101.pdf` lists two of them as
`body-060s-120dp000` and `body060s-120dp000`, without the `st`, while the other
four have it. That looks like a typo in the document — the files here follow the
consistent `body<offset>st<strike>dp<depth>` pattern. If the server's file list
disagrees, rename those two to match what the server shows.

## Node coverage of the contour files

Each contour file holds 44,551 nodes at 100 m spacing, covering `j` from -14900
to 14900 m and `k` from 100 to 14900 m, and every node ruptured (0.569 to 7.80 s
for TPV101, 0.569 to 7.80 s for TPV102). The specification allows `j` from
-15000 to 15000 and `k` from 0 to 15000, so the very edges of the fault are not
sampled; the server's Delaunay interpolation covers the interior, which is where
the contours are drawn.
