# Manuscript results and scripts

This folder holds the result files of the controlled study and the scripts that turn them into its tables and figures, in the main text (Sections 4–6) and in the supplement. They run without a GPU, a model or a benchmark download. The results of the reference pipeline (BOP leaderboard scores, runtimes and its ablations) come from the BOP evaluation server and timing runs and are not regenerated here.

## Contents

| Path | Contents |
| --- | --- |
| `data33_17/` | Result files of the controlled study (JSON) and the Markdown tables derived from them (`TABLES_*.md`). The scripts locate tables in these Markdown files by their headings, which are in Chinese; the cell values are what the scripts read. |
| `data33_18/` | Result files of the supplementary SAM 3 detection gate on XYZ-IBD validation frames. |
| `data33_19/` | Result files of the further experiments (Supplementary Section S7): other scores of the same best poses, native scores of GigaPose and SAM-6D, the inversion in bins, varied count ratios and a rejection step, with the readings fixed in advance (`S19_RULES.json`, `S19_PREDICT.json`). |
| `data33_20/` | Result files of the reversed-score analyses (Supplementary Section S8): ranking by the per-call score with its sign reversed and by a calibration fitted on a few labelled frames, the categories of the incorrect poses behind the reversal, and intervals that resample whole scenes, with the readings fixed in advance (`S20_RULES.json`). |
| `scripts/` | Table, figure and number-check scripts. |

File names keep the internal run identifiers (33-17, 33-18). In `data33_18/`, absolute paths of the machines on which the runs were made were removed from file references; nothing else was changed. `data/paper_results.json` at the repository root is extracted from these files, and its `source_sha256` lists their hashes.

Not included: the qualitative examples (Fig. 2 and the qualitative panels of the supplement), because they are rendered from benchmark images, and the per-frame predictions, which require the benchmark data to be useful.

## Regenerating tables and figures

The table scripts need only the Python standard library. The figure script needs NumPy, SciPy and Matplotlib (tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11).

```bash
python paper/scripts/make_tables_v3.py         # main tables  -> paper/tables/
python paper/scripts/make_supp_tables_v3.py    # supplement   -> paper/tables/
python paper/scripts/make_s18_table.py         # SAM 3 gate   -> paper/tables/supp_sam3.tex
FIGW=7.1 python paper/scripts/make_figures_v3.py   # figures  -> paper/figures/
FIGW=7.1 python paper/scripts/make_s19.py          # count-ratio figure and Section S7 tables
python paper/scripts/make_s20.py                   # Section S8 tables
```

`FIGW` is the text width in inches: 7.1 for the two-column journal layout and 5.45 for the single-column review layout. The figure script also rewrites `data33_17/derived_v3.json`, a summary of values used in the text; the regenerated file is identical to the committed one. `make_s19.py` and `make_s20.py` likewise rewrite `data33_19/derived_s19.json` and `data33_20/derived_s20.json`.

## Checking the numbers in the text

```bash
python paper/scripts/check_numbers_v3.py [MANUSCRIPT_ROOT]
```

The script recomputes the numbers quoted in the running text from the result files. Given the root of the manuscript source (the folder that contains `sec/*.tex`), it reports for each number whether it appears in the text. Without it, the script prints the recomputed values.
