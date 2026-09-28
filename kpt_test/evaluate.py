"""Validate submissions against the teacher's immutable prepared test bundle."""

from pathlib import Path

from . import __version__
from .io import read_json, read_records, index_records, file_hash
from .metrics import evaluate_sets
from .semantic import load_embeddings
from .taxonomy import Taxonomy


def load_predictions(path, gold, taxonomy, allow_empty=False):
    rows = read_records(path)
    for row in rows:
        if set(row) != {"qid", "labels"}:
            raise ValueError(f"{path}: prediction records must contain exactly qid and labels")
    predictions = index_records(rows, "labels", allow_empty)
    missing, extra = set(gold) - predictions.keys(), predictions.keys() - set(gold)
    if missing or extra:
        raise ValueError(f"{path}: qid mismatch; missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}")
    for qid, labels in predictions.items():
        taxonomy.validate_labels(labels, f"{path}: qid={qid}")
    return predictions


def evaluate_submission(data_dir, predictions_path, noisy_predictions_path=None, embeddings_path=None,
                        skip_semantic=False, allow_empty=False):
    root = Path(data_dir)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported test bundle schema")
    required = {"test.inputs.jsonl", "test.gold.jsonl", "test.noisy.inputs.jsonl", "taxonomy.json", "splits.json"}
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("Manifest is missing required test files")
    # Validate only named bundle files. Never follow paths supplied by a manifest.
    for filename in sorted(required):
        if file_hash(root / filename) != files[filename]:
            raise ValueError(f"Test bundle hash mismatch: {filename}")
    taxonomy = Taxonomy(read_json(root / "taxonomy.json"))
    if taxonomy.dataset != manifest.get("dataset") or taxonomy.digest != manifest.get("taxonomy_sha256"):
        raise ValueError("Manifest and taxonomy do not agree")
    gold = index_records(read_records(root / "test.gold.jsonl"), "labels")
    for qid, labels in gold.items():
        taxonomy.validate_labels(labels, f"Gold qid={qid}")
    for filename in ("test.inputs.jsonl", "test.noisy.inputs.jsonl"):
        if set(index_records(read_records(root / filename))) != set(gold):
            raise ValueError(f"Bundle qid mismatch: {filename}")
    predictions = load_predictions(predictions_path, gold, taxonomy, allow_empty)
    noisy = load_predictions(noisy_predictions_path, gold, taxonomy, allow_empty) if noisy_predictions_path else None
    if skip_semantic and embeddings_path:
        raise ValueError("--skip-semantic cannot be combined with --embeddings")
    if not skip_semantic and not embeddings_path:
        default_cache = root / "embeddings.json"
        if default_cache.exists():
            embeddings_path = default_cache
        else:
            raise ValueError("Semantic cache required: use --embeddings, build-embeddings, or explicitly --skip-semantic")
    scores = load_embeddings(embeddings_path, taxonomy) if embeddings_path else None
    metrics, details = evaluate_sets(gold, predictions, taxonomy, scores)
    robustness = {"status": "not_evaluated", "reason": "No noisy predictions supplied", "ratio": None}
    if noisy is not None:
        noisy_metrics, _ = evaluate_sets(gold, noisy, taxonomy)
        original_p, noisy_p = metrics["exact"]["precision"], noisy_metrics["exact"]["precision"]
        robustness = {"status": "ok" if original_p else "undefined_zero_baseline",
                      "ratio": noisy_p / original_p if original_p else None,
                      "original_precision": original_p, "noisy_precision": noisy_p,
                      "noisy_exact": noisy_metrics["exact"],
                      "noisy_strict_accuracy": noisy_metrics["strict_accuracy"],
                      "noisy_loose_accuracy": noisy_metrics["loose_accuracy"],
                      "protocol": manifest["noise"]}
    thresholds = {"strict_accuracy": {"minimum": 0.90, "actual": metrics["strict_accuracy"], "passed": metrics["strict_accuracy"] >= 0.90},
                  "loose_accuracy": {"minimum": 0.95, "actual": metrics["loose_accuracy"], "passed": metrics["loose_accuracy"] >= 0.95}}
    report = {"schema_version": 1, "evaluator_version": __version__, "dataset": taxonomy.dataset,
              "evaluation_complete": scores is not None and noisy is not None,
              "metrics": metrics, "interaction_robustness": robustness,
              "semantic_status": "ok" if scores else "explicitly_skipped",
              "semantic_resource": scores.metadata if scores else None,
              "requirements": {"thresholds": thresholds,
                               "accuracy_thresholds_passed": all(item["passed"] for item in thresholds.values()),
                               "nonempty_predictions": all(predictions.values()) and (noisy is None or all(noisy.values()))},
              "inputs": {"manifest_sha256": file_hash(root / "manifest.json"),
                         "taxonomy_sha256": taxonomy.digest, "predictions_sha256": file_hash(predictions_path),
                         "noisy_predictions_sha256": file_hash(noisy_predictions_path) if noisy_predictions_path else None,
                         "embeddings_sha256": file_hash(embeddings_path) if embeddings_path else None}}
    return report, details
