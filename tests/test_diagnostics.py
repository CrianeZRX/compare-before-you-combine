from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from compare6d.check import check, load_bop_errors, load_table
from compare6d.cli import audit, main
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


class CheckTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_example_table_is_read_as_reversed(self):
        result = check(load_table(self.ROOT / "examples" / "scores_table.csv"))
        self.assertEqual((result["wins"], result["pairs"], result["images_with_pairs"]), (1, 10, 7))
        self.assertEqual(result["reading"], "reversed")
        self.assertLess(result["sign_flip_p"], 0.05)
        self.assertLess(result["interval_95"][1], 0.5)
        self.assertEqual(main(["check", "--table", str(self.ROOT / "examples" / "scores_table.csv")]), 0)

    def test_readings_follow_the_interval(self):
        image = lambda i: ("", "1", str(i))
        good = [(image(i), "1", s, c) for i in range(20) for s, c in ((0.9, True), (0.1, False))]
        self.assertEqual(check(good)["reading"], "comparable")
        mixed = [(image(i), "1", s, c) for i in range(20)
                 for s, c in (((0.9, True), (0.1, False)) if i % 2 else ((0.1, True), (0.9, False)))]
        self.assertEqual(check(mixed)["reading"], "not_comparable")
        self.assertEqual(check(mixed, seed=3), check(mixed, seed=3))
        with self.assertRaises(ValueError):
            check([(image(0), "1", 0.5, True), (image(0), "2", 0.4, False)])

    def test_few_informative_images_give_no_direction(self):
        image = lambda i: ("", "1", str(i))
        rows = [(image(i), "1", s, c) for i in range(5) for s, c in ((0.1, True), (0.9, False))]
        rows += [(image(i), "1", 0.5, True) for i in range(5, 40)]  # images without a pair do not count
        result = check(rows)
        self.assertEqual((result["images_with_pairs"], result["accuracy"]), (5, 0.0))
        self.assertEqual(result["reading"], "too_few_images")
        self.assertEqual(result["sign_flip_p"], 0.0625)
        six = [(image(i), "1", s, c) for i in range(6) for s, c in ((0.1, True), (0.9, False))]
        self.assertEqual(check(six)["reading"], "reversed")
        one = [(image(0), "1", s, c) for s, c in ((0.1, True), (0.9, False), (0.8, False), (0.7, False))]
        self.assertEqual(check(one)["reading"], "too_few_images")

    def test_images_without_a_common_direction_give_no_direction(self):
        import random
        image = lambda i: ("", "1", str(i))
        definite = 0
        for draw in range(200):
            flip = random.Random(draw)
            rows = []
            for i in range(10):
                good, bad = (0.9, 0.1) if flip.random() < 0.5 else (0.1, 0.9)
                rows += [(image(i), "1", good, True), (image(i), "1", bad, False)]
            definite += check(rows, draws=200, seed=draw)["reading"] in ("reversed", "comparable")
        self.assertLessEqual(definite, 10)  # at most 5% of 200 draws

    def test_sign_flip_p_is_exact_for_small_sets(self):
        from itertools import product
        from compare6d.check import sign_flip_p
        values = [-1.0, -0.5, 2.0, -1.5, 0.5, -3.0, 1.0]
        observed = abs(sum(values))
        hits = sum(abs(sum(s * v for s, v in zip(signs, values))) >= observed - 1e-12
                   for signs in product((-1, 1), repeat=len(values)))
        self.assertAlmostEqual(sign_flip_p(values), hits / 2 ** len(values))
        self.assertEqual(sign_flip_p([0.0, 0.0]), 1.0)

    def test_ties_count_as_half_a_pair(self):
        image = lambda i: ("", "1", str(i))
        constant = [(image(i), "1", 0.5, c) for i in range(10) for c in (True, False)]
        result = check(constant)
        self.assertEqual((result["accuracy"], result["accuracy_strict"], result["tie_rate"]), (0.5, 0.0, 1.0))
        self.assertEqual(result["interval_95"], [0.5, 0.5])
        self.assertEqual(result["reading"], "no_information")
        half = [(image(i), "1", s, c) for i in range(20)
                for s, c in (((0.9, True), (0.1, False)) if i % 2 else ((0.5, True), (0.5, False)))]
        result = check(half)
        self.assertEqual((result["accuracy"], result["tie_rate"]), (0.75, 0.5))
        self.assertEqual(result["reading"], "comparable")

    def test_pairs_stay_within_one_image_and_object(self):
        rows = [(("", "1", "0"), "1", 0.9, True), (("", "1", "0"), "2", 0.1, False),
                (("", "1", "1"), "1", 0.2, False), (("", "1", "0"), "1", 0.3, False)]
        self.assertEqual(check(rows)["pairs"], 1)

    def test_bop_error_files(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "models_info.json").write_text(json.dumps({"3": {"diameter": 100.0}}))
            records = [{"im_id": 7, "obj_id": 3, "est_id": 0, "score": 0.2, "errors": {"0": [9.9], "1": [80.0]}},
                       {"im_id": 7, "obj_id": 3, "est_id": 1, "score": 0.8, "errors": {"0": [10.0], "1": [55.0]}}]
            (folder / "errors_000002.json").write_text(json.dumps(records))
            rows = load_bop_errors(folder, folder / "models_info.json")
            self.assertEqual([row[3] for row in rows], [True, False])
            self.assertEqual(rows[0][0], ("", "000002", "7"))
            self.assertEqual(len(load_bop_errors(folder, folder / "models_info.json", threshold=0.2)), 2)
            self.assertTrue(all(row[3] for row in load_bop_errors(folder, folder / "models_info.json", threshold=0.2)))
            self.assertEqual(main(["check", "--bop-errors", str(folder)]), 1)


if __name__ == "__main__":
    unittest.main()
