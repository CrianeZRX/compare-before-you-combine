# Cached-pose inputs

## Audit JSON

`compare6d audit path/to/input.json` accepts one top-level `targets` array. A target is an image–object pair.

| Key | Scope | Meaning |
| --- | --- | --- |
| `image_id` | Target | Unique image key within the input. Include the dataset or scene in the key when combining datasets. |
| `object_id` | Target | Object identifier. |
| `calls` | Target | One or more mask calls. |
| `mask_id` | Call | Identifier unique within a target. |
| `scores` | Call | At least two finite scores in the saved hypothesis order. |
| `correct` | Call | One Boolean per hypothesis. The manuscript uses symmetry-aware MSSD strictly below 0.1 times the object diameter. |
| `joint_score` | Call | Optional score of that call's per-call winner in the joint batch of all winners for the same target. Provide it for every call or omit it throughout the input. |

The best per-call pose is the first maximum in the saved order. Relative scores use the population standard deviation, stable softmax, and the margin over the strongest alternative. The audit fixes the selected pose when evaluating each score. Pairwise counts only use opposite-label pairs, pooled over targets by pair count. The result reports strict accuracy and accuracy with half credit for ties. The primary paper comparison uses a strict higher-score win. The relative-score control uses half credit for ties.

The included JSON example is synthetic. For real diagnostics, compute MSSD against the matching benchmark objects in the same length unit as their diameters, with the benchmark symmetry definitions and split annotations. Correctness labels are used by the audit only.

## Check: one score per output pose

`compare6d check` computes the cross-call pairwise accuracy of the score that a pipeline compares or submits. It needs one row per output pose, where each pose is the best pose of one scoring call (one mask), and it assumes that all rows of an image and object come from separate calls.

```bash
compare6d check --table examples/scores_table.csv
compare6d check --bop-errors path/to/error=mssd_ntop=-1 --models-info path/to/models/models_info.json
```

| Input | Format |
| --- | --- |
| `--table` | CSV with the columns `scene_id`, `im_id`, `obj_id`, `score` and `correct` (`1`/`0` or `true`/`false`). An optional `dataset` column keeps the images of several datasets apart. |
| `--bop-errors` | Folder with the `errors_<scene>.json` files that the BOP toolkit's `eval_calc_errors.py` writes for the error type `mssd`. Each record gives `im_id`, `obj_id`, `score` and the error to every annotated instance of the object. With `--models-info`, a pose counts as correct if its smallest error is strictly below `--threshold` (default 0.1) times the object diameter. |

A pair is two poses of the same image and object of which exactly one is correct. The accuracy is the share of pairs in which the correct pose has the higher score, with a tied pair counted as half (the area under the ROC curve restricted to these pairs, so that 50% means no information). The point estimate, the 95% interval and the reading all use this convention; `accuracy_strict` and `tie_rate` report the share with the strictly higher score and the share of tied pairs. The interval resamples images (2,000 draws, fixed seed). An image with at least one pair is informative. The output field `reading` is `no_information` if every pair is tied, otherwise `too_few_images` if fewer than six images are informative, `reversed` or `comparable` if a two-sided sign-flip test across the informative images gives p < 0.05 (`sign_flip_p`; each image keeps or reverses all its pairs with probability one half; exact up to 16 images, otherwise 20,000 random sign patterns) and the interval lies below or above 50%, and `not_comparable` otherwise. Six is the smallest number of images for which the exact two-sided test can fall below 0.05. Images in which no object has both a correct and an incorrect pose contribute no pairs, so the check needs frames in which the pipeline makes mistakes: label frames until at least ten are informative.

## Selection API

Pass one image at a time to `select_detections`. Each `Candidate` is the already chosen best pose of one mask. Its translation and object diameter are in metres. `detector_confidence=None` identifies an admitted supplementary mask. `detector_counts` is the per-object count of accepted detections.

| Variant | Candidate pool | Output rule |
| --- | --- | --- |
| B0 | Detector masks | One registered pose per detection, with detector confidence. |
| B1 | Detector masks | Joint ranking, then NMS, up to the detector count. |
| B2 | Detector masks | B1 with the maximum of detector and survivor counts. |
| B3 | Detector plus supplementary masks | Joint ranking, then NMS, up to the detector count. |
| B4 | Detector plus supplementary masks | B3 with the maximum of detector and survivor counts. |

Compute joint scores separately for the candidate pool used by each variant. In B1–B4, NMS keeps translations at least 0.15 times the object diameter apart. The selected outputs use depth consistency as confidence and are capped at `2*n_hat + 8` per object and 100 per image. The source pools are assumed to have already passed mask admission and identity assignment.
