# Bond Batch Analytics (CSV → CSV/XLSX)

A small Python CLI that reads a bonds CSV, runs per-bond analytics, and exports:
- `bond_statistics.csv` (full results)
- `bond_statistics.xlsx` (results + summary tab)
- `run_metadata.json` (run context + outputs)

Each execution writes into a **new timestamped run folder** so nothing gets overwritten.

## Features
- Batch bond calculations (via `src/batch.py -> run_batch`)
- Excel output with formatting (`Bonds` + `Summary` sheets)
- Portfolio-style summary (MV-weighted YTM/duration, total DV01, breakdowns)
- Timestamped output folders: `output/runs/<YYYYMMDD_HHMMSS_id>/`
- `run_metadata.json` saved even if the run fails

## Project layout
├─ main.py
├─ input/
│ └─ example_bonds.csv
└─ src/
└─ batch.py

## Outputs

After each run you’ll get:

output/
└─ runs/
   └─ 20251227_183455_a1b2c3d4/
      ├─ bond_statistics.csv
      ├─ bond_statistics.xlsx
      └─ run_metadata.json
