from dataclasses import replace
import json
from pathlib import Path
import unittest

from compare6d.cli import audit
from compare6d.ranking import best_index, correct_from_mssd, pairwise, relative_scores, score_jointly
from compare6d.selection import Candidate, select_detections, translation_nms
from compare6d.replay import validate


class RankingTests(unittest.TestCase):
    def test_relative_scores_use_the_nominated_pose(self):
        self.assertEqual(relative_scores([5, 5])["softmax"], 0.5)
        self.assertEqual(relative_scores([5, 5])["z_score"], 0.0)
        self.assertEqual(relative_scores([2, 7], winner=0)["margin"], -5)
        self.assertEqual(relative_scores([2, 7]), relative_scores([1002, 1007]))

    def test_pair_tie_conventions(self):
        counts = pairwise([2, 2, 1], [True, False, False])
        self.assertEqual((counts.wins, counts.ties, counts.pairs), (1, 1, 2))
        self.assertEqual(counts.accuracy, 0.5)
        self.assertEqual(counts.accuracy_half_credit, 0.75)
        self.assertIsNone(pairwise([2, 1], [True, True]).accuracy)

    def test_mssd_is_strict_and_winner_ties_are_stable(self):
        self.assertFalse(correct_from_mssd(1.0, 10))
        self.assertTrue(correct_from_mssd(0.999, 10))
        self.assertEqual(best_index([1, 2, 2]), 1)
        with self.assertRaises(ValueError):
            best_index([1, float("nan")])

    def test_joint_rescore_passes_only_the_original_winners(self):
        def scorer(poses):
            self.assertEqual(poses, ["mask 1 winner", "mask 2 winner"])
            return [0.1, 0.9]

        result = score_jointly([["a", "mask 1 winner"], ["mask 2 winner", "b"]],
                               [[-1, 100], [6, 3]], scorer)
        self.assertEqual(result, (0.1, 0.9))
        with self.assertRaises(ValueError):
            score_jointly([[1]], [[0]], lambda _: [])

    def test_synthetic_audit_recovers_inversion(self):
        doc = json.loads(Path("examples/synthetic_calls.json").read_text())
        result = audit(doc)
        self.assertEqual(result["cross_call"]["per_call"]["accuracy"], 0.0)
        self.assertEqual(result["cross_call"]["joint"]["accuracy"], 1.0)
        self.assertEqual(result["within_call_per_call_score"]["accuracy"], 1.0)
        del doc["targets"][0]["calls"][0]["joint_score"]
        with self.assertRaises(ValueError):
            audit(doc)


class SelectionTests(unittest.TestCase):
    def test_paper_detection_arms_keep_their_quota_and_pool(self):
        base = Candidate(1, "detector", (0, 0, 0), 0.1, 0.9, 0.8)
        supplemented = Candidate(1, "supplement", (0.4, 0, 0), 0.9, 0.8)
        near = replace(supplemented, mask_id="near", translation_m=(0.45, 0, 0),
                       joint_score=0.7)
        unseen = replace(supplemented, object_id=2)
        rows = [base, supplemented, near, unseen]
        expected = {"B0": ["detector"], "B1": ["detector"], "B2": ["detector"],
                    "B3": ["supplement"], "B4": ["detector", "supplement"]}
        for variant, masks in expected.items():
            with self.subTest(variant=variant):
                selected = select_detections(rows, {1: 1, 2: 0}, {1: 1.0}, variant)
                self.assertEqual([r.mask_id for r in selected], masks)

    def test_nms_keeps_exact_boundary(self):
        first = Candidate(1, "a", (0, 0, 0), 1, 1, 1)
        second = replace(first, mask_id="b", translation_m=(0.15, 0, 0))
        self.assertEqual(translation_nms([first, second], 0.15), [first, second])

    def test_count_caps_apply_per_object_and_per_image(self):
        rows = [Candidate(obj, str(i), (float(i), 0, 0), -i, 1.0)
                for obj in range(1, 13) for i in range(15)]
        counts = {obj: 1 for obj in range(1, 13)}
        diameters = {obj: 1.0 for obj in counts}
        result = select_detections(rows, counts, diameters, "B4")
        self.assertEqual(len(result), 100)
        self.assertEqual(sum(row.object_id == 1 for row in result), 10)


class EvidenceTests(unittest.TestCase):
    def test_archived_reference_values_and_validation(self):
        data = json.loads(Path("data/paper_results.json").read_text())
        validate(data)
        pooled = data["cross_call_pooled"]
        self.assertEqual(pooled["raw"]["opposite_label_pairs"], 36852)
        self.assertAlmostEqual(pooled["raw"]["accuracy"], 0.0517, places=4)
        self.assertAlmostEqual(pooled["joint"]["accuracy"], 0.9397, places=4)
        data["cross_call_pooled"]["raw"]["correct_first"] += 1
        with self.assertRaises(ValueError):
            validate(data)


if __name__ == "__main__":
    unittest.main()
