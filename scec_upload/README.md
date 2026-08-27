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

Both passed `validate_scec_dump.py`, which checks the header keys, the field-list
line, the column count, the uniform time step, and the initial conditions.

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

## Two things to check before uploading

**The `author` field** currently reads `author=Tatva validation workflow` in every
file. For a real submission this should be your name, and probably your
affiliation. To rewrite it across the set:

```bash
sed -i 's|^# author=.*|# author=Your Name, National Taiwan University|' scec_upload/*/*.txt
```

**The off-fault file names.** `uploadTPV101.pdf` lists two of them as
`body-060s-120dp000` and `body060s-120dp000`, without the `st`, while the other
four have it. That looks like a typo in the document — the files here follow the
consistent `body<offset>st<strike>dp<depth>` pattern. If the server's file list
disagrees, rename those two to match what the server shows.
