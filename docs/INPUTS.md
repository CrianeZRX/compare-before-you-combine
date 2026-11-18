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
