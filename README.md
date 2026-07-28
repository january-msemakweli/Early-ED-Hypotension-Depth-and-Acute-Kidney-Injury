# Depth of Early ED Hypotension and Acute Kidney Injury

Analysis code for a target-trial-style observational study of early emergency
department hypotension depth and 7-day acute kidney injury or death. Among
adults with a first new sustained ED hypotensive episode in MIMIC-IV-ED
(v2.2) linked to MIMIC-IV (v3.1), the primary exposure is the MAP nadir over
the following two hours, contrasted as deep versus shallow tertiles of depth,
with charted minutes below 65 mmHg treated as a measurement comparison rather
than a competing exposure.

## Data availability

This repository contains **no patient-level data**. MIMIC-IV and MIMIC-IV-ED
are available to credentialed researchers from PhysioNet
(https://physionet.org/content/mimiciv/ and
https://physionet.org/content/mimic-iv-ed/) and cannot be redistributed under
the data use agreement. The `tables/cache/` directory (episode-level
intermediates and lab screens) and `tables/analytic_cohort.csv` are
patient-level and are intentionally excluded by `.gitignore`. Only code and
non-identifiable aggregate outputs (cohort-level tables and the published
figures) are included.

## Repository layout

```
scripts/            Python analysis pipeline
  config.py         Shared paths and pre-specified constants
  analysis.py       End-to-end cohort build, estimation, tables, figures
  rebuild_figures.py  Regenerate figures from saved aggregate tables
  make_notebook.py  Optional: convert analysis.py to a notebook
  power_calc.py     Fixed-cohort precision statement
tables/             Non-identifiable aggregate outputs
figures/            Publication figures (PDF, PNG, TIFF)
requirements.txt    Pinned Python dependencies
```

## Reproducing the analysis

1. Obtain MIMIC-IV-ED v2.2 and MIMIC-IV v3.1 from PhysioNet and place the
   gzipped tables so that one of the following roots exists (see
   `scripts/config.py`):

```
../physionet.org/files/mimic-iv-ed/2.2/ed/
../physionet.org/files/mimiciv/3.1/hosp/
../physionet.org/files/mimiciv/3.1/icu/
```

   Adjust `DATA_ROOT` in `scripts/config.py` if your layout differs.

2. Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run from the repository root:

```bash
python scripts/analysis.py
```

   To regenerate figures from the saved aggregate tables without re-running
   the full pipeline:

```bash
python scripts/rebuild_figures.py
```

Patient-level intermediates are written to `tables/cache/` (git-ignored);
aggregate tables and figures are written to `tables/` and `figures/`.

## License

Code is released under the MIT License (see `LICENSE`). MIMIC-IV and
MIMIC-IV-ED data are governed separately by the PhysioNet data use agreement.
