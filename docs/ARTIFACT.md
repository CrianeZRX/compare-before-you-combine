# Paper artifact

| Paper topic | Included here |
| --- | --- |
| Cross-call ranking and set dependence | Cached-pose Python API, synthetic test case, original per-dataset and pooled aggregate counts, localization AR and archived paired contrasts. |
| Coverage versus candidate budget | Archived density analysis and eight-slot statistics. The density analysis using joint scores as its basis is labeled post hoc. |
| Pose-assisted counting | B0–B4 selection code, dataset-level AP, paired intervals, and output counts and composition. |
| SAM 3 supplementary control | Reported 2D gate metrics for the XYZ-IBD check split. |
| CoVer6D reference results | Described in the manuscript. This project's executable selection API implements the controlled detection study. |

The localization results cover the LM-O, T-LESS, TUD-L, IC-BIN, and YCB-V benchmark test sets. Their reported mean weights datasets equally. The detection results cover 150 XYZ-IBD validation frames and 88 IPD validation frames. IPD was processed at half resolution. AR and AP are reported in percentage points. Pair accuracies and coverage ratios in JSON and CSV are fractions.

## Evidence provenance

`data/paper_results.json` contains the selected aggregate evidence. Its `source_sha256` lists the SHA-256 of the result files in `paper/` from which it was extracted, so the extraction can be repeated from the files in this repository.

```bash
python tools/export_aggregates.py paper /tmp/paper_results.json
python tools/make_overview.py data/paper_results.json /tmp/results-overview.png
```

The second command requires Matplotlib. The README figure uses only the archived aggregate counts and estimates.

The portable implementation uses the formulas and protocols described in the manuscript and the archived relative-score experiment. The replay reads the recorded registrations, intervals and BOP metrics as summarized aggregates. Fresh prediction-level measurements require the external model and benchmark inputs. The released interfaces consume registered hypotheses or cached best poses to separate post-registration diagnostics from the model and dataset installation.

## External components

The study used FoundationPose and MegaPose scoring, CNOS and NOCTIS detections, SAM2 masks, and DINOv2 features. Benchmark data and annotations come from BOP. Use the upstream distributions and their terms when connecting a real scorer or dataset. This artifact contains the authors' post-registration algorithms, the result files and table and figure scripts of the manuscript, the recorded aggregates, and original synthetic data. It contains no third-party code, weights, images or meshes.
