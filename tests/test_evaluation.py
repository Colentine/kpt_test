import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kpt_test.adapters import load_da20k, load_xes3g5m
from kpt_test.cli import main
from kpt_test.evaluate import evaluate_submission
from kpt_test.io import read_json, read_records, write_json, write_jsonl, index_records
from kpt_test.metrics import evaluate_sets, optimal_match
from kpt_test.prepare import prepare_dataset, perturb_text, split_examples
from kpt_test.semantic import SemanticScores
from kpt_test.taxonomy import Taxonomy


def taxonomy_fixture():
    paths = {"a": ["r", "g1", "p1", "a"], "b": ["r", "g1", "p1", "b"],
             "c": ["r", "g1", "p2", "c"], "d": ["r", "g2", "p3", "d"], "e": ["s", "e"]}
    return Taxonomy({"dataset": "da20k", "labels": [
        {"id": kid, "name": kid, "path": path, "path_names": path}
        for kid, path in paths.items()]})


def cache_fixture(taxonomy):
    return {"schema_version": 1, "taxonomy_sha256": taxonomy.digest, "model": "synthetic-unit-test",
            "vectors": {kid: vector.tolist() for kid, vector in zip(taxonomy.ids, np.eye(len(taxonomy.ids)))}}


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = taxonomy_fixture()

    def test_hand_calculated_multi_label_metrics(self):
        gold = {"1": ["a", "b"], "2": ["c"], "3": ["a"]}
        pred = {"1": ["a", "c"], "2": ["c"], "3": ["b"]}
        metrics, _ = evaluate_sets(gold, pred, self.taxonomy)
        self.assertEqual(metrics["strict_accuracy"], 1 / 3)
        self.assertEqual(metrics["loose_accuracy"], 2 / 3)
        self.assertEqual(metrics["exact"]["tp"], 2)
        self.assertEqual(metrics["exact"]["precision"], .5)
        self.assertEqual(metrics["exact"]["recall"], .5)
        self.assertAlmostEqual(metrics["exact"]["macro_f1"], 4 / 15)
        self.assertAlmostEqual(metrics["hierarchical"]["tp"], 2.7)

    def test_each_hierarchical_tier(self):
        for truth, expected in {"a": 1, "b": .5, "c": .2, "d": .05, "e": 0}.items():
            with self.subTest(truth=truth):
                self.assertEqual(self.taxonomy.hierarchy_score("a", truth), expected)

    def test_optimal_assignment_beats_greedy(self):
        matrix = [[.9, .8], [.85, 0]]
        self.assertAlmostEqual(optimal_match([0, 1], [0, 1], lambda p, t: matrix[p][t]), 1.65)

    def test_assignment_stays_optimal_above_twenty_labels(self):
        matrix = np.eye(22)
        matrix[:2, :2] = [[.9, .8], [.85, 0]]
        self.assertAlmostEqual(optimal_match(list(range(22)), list(range(22)), lambda p, t: matrix[p, t]), 21.65)

    def test_no_repeated_credit_for_one_true_label(self):
        metrics, _ = evaluate_sets({"q": ["a"]}, {"q": ["a", "b"]}, self.taxonomy)
        self.assertEqual(metrics["hierarchical"]["tp"], 1)
        self.assertEqual(metrics["hierarchical"]["precision"], .5)

    def test_abstention_is_zero(self):
        metrics, _ = evaluate_sets({"q": ["a"]}, {"q": []}, self.taxonomy)
        self.assertEqual(metrics["exact"]["f1"], 0)
        self.assertEqual(metrics["hierarchical"]["precision"], 0)

    def test_semantic_cosine_clipped_and_normalized(self):
        payload = cache_fixture(self.taxonomy)
        payload["vectors"]["b"] = [-2, 0, 0, 0, 0]
        payload["vectors"]["c"] = [3, 4, 0, 0, 0]
        scores = SemanticScores(payload, self.taxonomy)
        self.assertEqual(scores("a", "b"), 0)
        self.assertAlmostEqual(scores("a", "c"), .6)
        self.assertEqual(scores("a", "a"), 1)

    def test_invalid_semantic_caches(self):
        for invalid in ([], [0] * 5, [float("nan")] * 5, [float("inf")] * 5, [1, 2], [True] * 5):
            with self.subTest(invalid=invalid):
                payload = cache_fixture(self.taxonomy)
                payload["vectors"]["a"] = invalid
                with self.assertRaises(ValueError):
                    SemanticScores(payload, self.taxonomy)
        payload = cache_fixture(self.taxonomy)
        payload["taxonomy_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            SemanticScores(payload, self.taxonomy)

    def test_invalid_taxonomy_nodes(self):
        for change in (lambda rows: rows.append(rows[0]),
                       lambda rows: rows[1].update(path=["s", "p1", "b"], path_names=["s", "p1", "b"]),
                       lambda rows: rows[0].update(path=["r", "r", "a"], path_names=["r", "r", "a"])):
            data = json.loads(json.dumps(self.taxonomy.data))
            change(data["labels"])
            with self.assertRaises(ValueError):
                Taxonomy(data)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.taxonomy = taxonomy_fixture()
        self.examples = [{"qid": str(i), "text": f"Question number {i}: solve for x", "labels": ["a"]} for i in range(20)]
        self.bundle = self.root / "bundle"
        self.manifest = prepare_dataset(self.examples, self.taxonomy, self.bundle)
        self.gold = read_records(self.bundle / "test.gold.jsonl")
        self.pred = self.root / "pred.jsonl"
        self.noisy = self.root / "noisy.jsonl"
        write_jsonl(self.pred, list(reversed(self.gold)))
        write_jsonl(self.noisy, self.gold)
        write_json(self.bundle / "embeddings.json", cache_fixture(self.taxonomy))

    def test_complete_offline_evaluation(self):
        report, details = evaluate_submission(self.bundle, self.pred, self.noisy)
        self.assertTrue(report["evaluation_complete"])
        self.assertTrue(report["requirements"]["accuracy_thresholds_passed"])
        self.assertEqual(report["metrics"]["strict_accuracy"], 1)
        self.assertEqual(report["metrics"]["semantic"]["precision"], 1)
        self.assertEqual(report["interaction_robustness"]["ratio"], 1)
        self.assertEqual(len(details), 2)

    def test_partial_evaluation_is_explicit(self):
        report, _ = evaluate_submission(self.bundle, self.pred, skip_semantic=True)
        self.assertFalse(report["evaluation_complete"])
        self.assertIsNone(report["metrics"]["semantic"])
        self.assertIsNone(report["interaction_robustness"]["ratio"])

    def test_zero_baseline_robustness_is_null(self):
        write_jsonl(self.pred, [{"qid": row["qid"], "labels": ["b"]} for row in self.gold])
        report, _ = evaluate_submission(self.bundle, self.pred, self.noisy)
        self.assertIsNone(report["interaction_robustness"]["ratio"])
        self.assertEqual(report["interaction_robustness"]["status"], "undefined_zero_baseline")
        self.assertFalse(report["requirements"]["accuracy_thresholds_passed"])

    def test_robustness_can_exceed_one(self):
        rows = [dict(row) for row in self.gold]
        rows[0] = {"qid": rows[0]["qid"], "labels": ["b"]}
        write_jsonl(self.pred, rows)
        report, _ = evaluate_submission(self.bundle, self.pred, self.noisy)
        self.assertEqual(report["interaction_robustness"]["ratio"], 2)

    def test_bad_predictions_rejected(self):
        qid = self.gold[0]["qid"]
        cases = [[], self.gold[:1], self.gold + self.gold[:1],
                 self.gold + [{"qid": "extra", "labels": ["a"]}],
                 [{"qid": qid, "labels": ["outside"]}] + self.gold[1:],
                 [{"qid": qid, "labels": ["a", "a"]}] + self.gold[1:],
                 [{"qid": qid, "labels": "a"}] + self.gold[1:],
                 [{"qid": qid, "labels": []}] + self.gold[1:],
                 [{"qid": qid, "labels": ["a"], "gold": ["a"]}] + self.gold[1:]]
        for rows in cases:
            with self.subTest(rows=rows):
                write_jsonl(self.pred, rows)
                with self.assertRaises(ValueError):
                    evaluate_submission(self.bundle, self.pred, skip_semantic=True)

    def test_empty_predictions_need_explicit_flag(self):
        write_jsonl(self.pred, [{"qid": row["qid"], "labels": []} for row in self.gold])
        report, _ = evaluate_submission(self.bundle, self.pred, skip_semantic=True, allow_empty=True)
        self.assertEqual(report["metrics"]["exact"]["precision"], 0)
        self.assertFalse(report["requirements"]["nonempty_predictions"])

    def test_modified_bundle_is_rejected(self):
        write_jsonl(self.bundle / "test.gold.jsonl", self.gold[:1])
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            evaluate_submission(self.bundle, self.pred)

    def test_split_and_noise_stable_under_source_reordering(self):
        other = self.root / "other"
        manifest = prepare_dataset(list(reversed(self.examples)), self.taxonomy, other)
        self.assertEqual(self.manifest, manifest)
        splits = read_json(other / "splits.json")
        self.assertEqual([len(splits[k]) for k in ("train", "valid", "test")], [16, 2, 2])
        self.assertFalse(set(splits["train"]) & set(splits["test"]))

    def test_inputs_have_no_labels_or_answers(self):
        for filename in ("test.inputs.jsonl", "test.noisy.inputs.jsonl"):
            self.assertTrue(all(set(row) == {"qid", "text"} for row in read_records(self.bundle / filename)))

    def test_supplied_splits_reject_overlap_and_missing(self):
        path = self.root / "splits.json"
        for obj in ({"train": ["0"], "valid": [], "test": ["0"]},
                    {"train": [], "valid": [], "test": ["0"]}):
            write_json(path, obj)
            with self.assertRaises(ValueError):
                split_examples(self.examples, 42, path)

    def test_cli_output_and_exit_codes(self):
        args = ["evaluate", "--data-dir", str(self.bundle), "--predictions", str(self.pred),
                "--noisy-predictions", str(self.noisy), "--output", str(self.root / "report.json"),
                "--details", str(self.root / "details.jsonl"), "--require-complete", "--fail-on-threshold"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args + ["--skip-semantic"]), 1)
            self.assertEqual(main(args + ["--output", str(self.pred)]), 2)
        self.assertEqual(len(read_records(self.root / "details.jsonl")), 2)

    def test_duplicate_json_keys_rejected(self):
        self.pred.write_text('{"qid":"1","qid":"2","labels":["a"]}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            read_records(self.pred)

    def test_json_files_have_portable_lf_line_endings(self):
        path = self.root / "portable.json"
        write_json(path, {"text": "中文", "items": [1, 2]})
        self.assertNotIn(b"\r\n", path.read_bytes())
        self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_duplicate_integer_and_string_ids_rejected(self):
        with self.assertRaises(ValueError):
            index_records([{"qid": 1}, {"qid": "1"}])

    def test_noise_boundaries_and_reproducibility(self):
        text = "题目：设 x+1=3，求 x 的值。"
        self.assertEqual(perturb_text(text, random.Random(1), 0, 0), text)
        self.assertEqual(perturb_text("1234", random.Random(1), 1, 1), "1234")
        self.assertEqual(perturb_text(text, random.Random(1)), perturb_text(text, random.Random(1)))
        self.assertTrue(perturb_text(text, random.Random(1), 1, 1))
        with self.assertRaises(ValueError):
            perturb_text(text, random.Random(1), 1.1, 0)

    def test_accuracy_threshold_boundaries(self):
        splits = self.root / "all_test.json"
        write_json(splits, {"train": [], "valid": [], "test": [row["qid"] for row in self.examples]})
        bundle = self.root / "all_test"
        prepare_dataset(self.examples, self.taxonomy, bundle, split_file=splits)
        rows = [{"qid": row["qid"], "labels": ["a"]} for row in self.examples]
        rows[-2]["labels"] = ["a", "b"]
        rows[-1]["labels"] = ["b"]
        write_jsonl(self.pred, rows)
        report, _ = evaluate_submission(bundle, self.pred, skip_semantic=True)
        self.assertEqual(report["metrics"]["strict_accuracy"], .9)
        self.assertEqual(report["metrics"]["loose_accuracy"], .95)
        self.assertTrue(report["requirements"]["accuracy_thresholds_passed"])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions, self.labels, self.links = [self.root / name for name in ("questions.json", "labels.json", "links.json")]

    def test_da20k_official_parent_labels_and_inline_questions(self):
        write_json(self.labels, [{"label_id": 3, "name": "集合的含义", "label_father_id": "f0", "label_father_name": "集合"}])
        write_json(self.questions, [{"id": 1, "content": "集合的元素有几个？", "labels": [3], "analysis": "DO NOT LEAK", "options": {"A": "1", "B": "2"}}])
        rows, taxonomy, _ = load_da20k(self.questions, self.labels)
        self.assertEqual(rows[0]["labels"], ["3"])
        self.assertNotIn("DO NOT LEAK", rows[0]["text"])
        self.assertIn("B: 2", rows[0]["text"])
        self.assertEqual(taxonomy.labels["3"]["path"], ["f0", "3"])

    def test_da20k_sgpe_uuid_tree_and_separate_tags(self):
        write_json(self.labels, [{"id": 1, "uuid": "root", "name": "集合", "parent_uuid": ""},
                                 {"id": 3, "uuid": "leaf", "name": "子集", "parent_uuid": "root"}])
        write_json(self.questions, [{"id": 10, "text_processed": "判断两个集合的关系"}])
        write_json(self.links, [{"qid": 10, "label_id": 1}, {"qid": 10, "label_id": 3}, {"qid": 10, "label_id": 3}])
        rows, taxonomy, info = load_da20k(self.questions, self.labels, self.links)
        self.assertEqual(rows[0]["labels"], ["3"])
        self.assertEqual(info["removed_redundant_ancestor_tags"], 1)
        self.assertEqual(taxonomy.labels["3"]["path"], ["1", "3"])

    def test_da_tree_cycle_rejected(self):
        write_json(self.labels, [{"id": 1, "name": "a", "parent_id": 2}, {"id": 2, "name": "b", "parent_id": 1}])
        with self.assertRaisesRegex(ValueError, "cycle"):
            load_da20k(self.questions, self.labels)

    def test_xes_official_metadata(self):
        write_json(self.labels, {"0": "数学----运算----加法", "1": "数学----运算----减法"})
        write_json(self.questions, {"001": {"content": "计算 3+2-1", "kc_routes": ["数学----运算----加法", "数学----运算----减法"],
                                             "options": None, "answer": "4", "analysis": "DO NOT LEAK"}})
        rows, taxonomy, _ = load_xes3g5m(self.questions, self.labels)
        self.assertEqual(rows[0]["qid"], "001")
        self.assertEqual(rows[0]["labels"], ["0", "1"])
        self.assertEqual(rows[0]["text"], "计算 3+2-1")
        self.assertEqual(taxonomy.hierarchy_score("0", "1"), .5)

    def test_xes_same_names_in_different_branches(self):
        write_json(self.labels, {"0": ["甲", "计算", "应用"], "1": ["乙", "计算", "应用"]})
        write_json(self.questions, {"1": {"content": "一道测试题", "kc_routes": [["甲", "计算", "应用"]]}})
        _, taxonomy, _ = load_xes3g5m(self.questions, self.labels)
        self.assertEqual(taxonomy.hierarchy_score("0", "1"), 0)
        write_json(self.questions, {"1": {"content": "一道测试题", "kc_routes": ["应用"]}})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            load_xes3g5m(self.questions, self.labels)

    def test_xes_removes_only_redundant_internal_kcs(self):
        write_json(self.labels, {"0": "数学/运算", "1": "数学/运算/加法"})
        write_json(self.questions, {"1": {"content": "计算 1+1", "kc_routes": ["数学/运算", "数学/运算/加法"]}})
        rows, taxonomy, info = load_xes3g5m(self.questions, self.labels)
        self.assertEqual(taxonomy.ids, ["1"])
        self.assertEqual(rows[0]["labels"], ["1"])
        self.assertEqual(info["removed_redundant_ancestor_tags"], 1)


class ShippedExamplesTests(unittest.TestCase):
    def test_both_dataset_examples_have_hand_calculated_results(self):
        base = Path(__file__).resolve().parents[1] / "examples"
        for dataset in ("da20k", "xes3g5m"):
            with self.subTest(dataset=dataset):
                report, _ = evaluate_submission(base / dataset, base / f"{dataset}.predictions.jsonl",
                                               base / f"{dataset}.noisy.predictions.jsonl")
                metrics = report["metrics"]
                self.assertEqual(metrics["strict_accuracy"], .5)
                self.assertEqual(metrics["loose_accuracy"], .75)
                self.assertEqual(metrics["exact"]["micro_f1"], .6)
                self.assertAlmostEqual(metrics["exact"]["macro_f1"], 7 / 15)
                self.assertAlmostEqual(metrics["hierarchical"]["f1"], .68)
                self.assertAlmostEqual(metrics["semantic"]["f1"], .9114170815091945, places=7)
                self.assertTrue(report["evaluation_complete"])
                self.assertEqual(report["interaction_robustness"]["ratio"], 1.25)


if __name__ == "__main__":
    unittest.main()
