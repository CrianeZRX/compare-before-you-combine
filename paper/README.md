# Manuscript results and scripts

This folder holds the result files of the controlled study and the scripts that turn them into its tables and figures, in the main text (Sections 4–6) and in the supplement. They run without a GPU, a model or a benchmark download. The results of the reference pipeline (BOP leaderboard scores, runtimes and its ablations) come from the BOP evaluation server and timing runs and are not regenerated here.

## Contents

| Path | Contents |
| --- | --- |
| `data33_17/` | Result files of the controlled study (JSON) and the Markdown tables derived from them (`TABLES_*.md`). The scripts locate tables in these Markdown files by their headings, which are in Chinese; the cell values are what the scripts read. |
| `data33_18/` | Result files of the supplementary SAM 3 detection gate on XYZ-IBD validation frames. |
| `data33_19/` | Result files of the further experiments (Supplementary Section S2): other scores of the same best poses, native scores of GigaPose and SAM-6D, the inversion in bins, varied count ratios and a rejection step, with the readings fixed in advance (`S19_RULES.json`, `S19_PREDICT.json`). |
| `data33_20/` | Result files of the reversed-score analyses (Supplementary Sections S1 and S2): ranking by the per-call score with its sign reversed and by a calibration fitted on a few labelled frames, the categories of the incorrect poses behind the reversal, and intervals that resample whole scenes, with the readings fixed in advance (`S20_RULES.json`). |
| `data33_21/` | Result files of two further checks (Supplementary Sections S2 and S4): the count ratio of the detector on two more dense datasets (HomebrewedDB, HOPE-Image) at its official threshold, with a control frame of IC-BIN, and a small pilot scorer trained to rank the hypotheses of one call, with the readings fixed in advance (`S21_RULES.json`, `S21_RULES_AMEND.json`, `S21_E2_RULES.json`). |
| `data33_22/` | Result files of the experiments on the uses of a pose score: FoundationPose and other scores as output confidences (`S22_E1.json`) and as grasp order (`S22_E2.json`), the controlled displacement of ground-truth poses (`S22_E3.json`), and the check of the scorer of the released WAPR code (`S22_E4.json`, 57 MB; `s22_wbps_wrapper.py`), with the readings fixed in advance (`S22_RULES.json`, `S22_RULES_AMEND.json`). |
| `data33_23/` | Result files of the WBPS, public-file and interval experiments (Supplementary Sections S1 to S3): WBPS and other scores as confidence and grasp order on WAPR poses (`S23_B.json`), the check on public BOP submission files with only the confidence replaced (`S23_A.json`, `S23_A0_FEASIBILITY.json`; the downloaded files are not included, only their addresses and SHA-256 sums), and the BCa recomputation of earlier intervals with the comparison of the 750-image sample to the full test sets (`S23_C.json`), with the criteria fixed in advance (`S23_RULES.json`, `S23_RULES_AMEND.json`). |
| `data33_24/` | Result files of the CPU-only analyses of round 33-24: the replay of the check, before its six-image rule, on 3, 5, 10 and 20 random labelled frames per dataset, on the best poses of WAPR (`S24_B_wapr.json`, `S24_A_wapr.json`, run by `replay.py` from `S22_E4.json` under `S24_RULES.json`) and of FoundationPose (`S24_B_FP.json`, with the per-pose rows in `S24_B_rows/`); paired AP differences of the main contrasts with image, percentile, scene and jackknife intervals (`S24_C.json`); and the re-check of earlier pairwise accuracies with ties counted half (`S24_A_rest.json`). Criteria fixed in advance: `S24_RULES.json`, `S24_RULES_SERVER.json`. |
| `data33_25/` | The released check with its six-image rule (`tool_check_v1_1.py`, identical to `src/compare6d/check.py`): a random-sign control in which each drawn frame keeps or negates all its scores (`S25_null.json`, `S25_seqnull.json`), the replay of `data33_24` with the same draws (`S25_replay.json`), frames labelled until six or ten hold a pair (`S25_seq.json`) and all frames (`S25_full.json`), all run by `run25.py` on the rows of `data33_24`; and the same check on the public submission files of `data33_23` (`S25_PUBLIC.json`, report `REPORT33_25_S.md`; nine files whose per-pose labels were not archived are listed as pending). Criteria fixed in advance: `S25_RULES.json`, `S25_RULES_AMEND.json`, `S25_PUBLIC_RULES.json`. |
| `data_survey/` | What the method pages of two BOP leaderboards state about the use of FoundationPose, transcribed by hand on 2026-11-19 (Supplementary Section S3), and the counts derived from it. |
| `scripts/` | Table, figure and number-check scripts. |

File names keep the internal run identifiers (33-17, 33-18). In `data33_18/`, absolute paths of the machines on which the runs were made were removed from file references; nothing else was changed. `data/paper_results.json` at the repository root is extracted from these files, and its `source_sha256` lists their hashes.

Not included: the qualitative examples (Fig. 2 and the qualitative panels of the supplement), because they are rendered from benchmark images, and the per-frame predictions, which require the benchmark data to be useful.

## Regenerating tables and figures

The table scripts need only the Python standard library. The figure script needs NumPy, SciPy and Matplotlib (tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11).

```bash
python paper/scripts/make_tables_v3.py         # main tables  -> paper/tables/
python paper/scripts/make_supp_tables_v3.py    # supplement   -> paper/tables/
python paper/scripts/make_s18_table.py         # SAM 3 gate table (not used in this version)
FIGW=7.1 python paper/scripts/make_figures_v3.py   # figures  -> paper/figures/
FIGW=7.1 python paper/scripts/make_s19.py          # Supplementary Section S2 tables (its figure is not used)
python paper/scripts/make_s20.py                   # Supplementary Sections S1 and S2 tables
python paper/scripts/make_s21.py                   # count-ratio table of Supplementary Section S4
python paper/scripts/make_usage.py                 # Supplementary Section S3 table
python paper/scripts/make_s22.py                   # Table 2, its supplementary tables and the numbers of Figure 2
python paper/scripts/make_s23.py                   # Supplementary Tables S5 and S27 (S27 also feeds Figure 11)
python paper/scripts/make_main_moved.py            # Table 4, derived from a table of make_s23.py
python paper/scripts/make_s24.py                   # the AP-difference tables of the supplement
python paper/scripts/make_s25.py                   # Table 6 and the replay, random-sign and labelling tables of the supplement
python paper/scripts/make_figs_v4.py               # all data figures of the article (1, 3-7, 9-11) in one style; run it last,
                                                   # since make_figures_v3.py also writes Figures 1, 3 and 6 in the old style.
                                                   # Also reads tab_ladder.tex and main_mean_time.csv (hand-checked manuscript inputs)
```

`FIGW` is the text width in inches: 7.1 for the two-column journal layout and 5.45 for the single-column review layout. The figure script also rewrites `data33_17/derived_v3.json`, a summary of values used in the text; the regenerated file is identical to the committed one. `make_s19.py`, `make_s20.py` and `make_s21.py` likewise rewrite `data33_19/derived_s19.json`, `data33_20/derived_s20.json` and `data33_21/derived_s21.json`. `make_usage.py` rewrites `data_survey/derived_usage.json`. `make_s22.py` rewrites `data33_22/derived_s22.json`, `make_s23.py` rewrites `data33_23/derived_s23.json` (it uses SciPy for one rank correlation if available), `make_s24.py` rewrites `data33_24/derived_s24.json`, and `make_s25.py` rewrites `data33_25/derived_s25.json`. All pairwise accuracies count a tie as half a pair.

## Checking the numbers in the text

```bash
python paper/scripts/check_numbers_v3.py [MANUSCRIPT_ROOT]
```

The script recomputes the numbers quoted in the running text from the result files. Given the root of the manuscript source (the folder that contains `sec/*.tex`), it reports for each number whether it appears in the text. Without it, the script prints the recomputed values.
