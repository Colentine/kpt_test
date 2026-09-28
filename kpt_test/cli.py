"""Command-line interfaces for data preparation and evaluation."""

import argparse
import json
import sys
from pathlib import Path

from .adapters import load_da20k, load_xes3g5m
from .evaluate import evaluate_submission
from .io import read_json, write_json, write_jsonl
from .prepare import prepare_dataset
from .semantic import DEFAULT_MODEL, build_embeddings
from .taxonomy import Taxonomy


def dataset_name(value):
    normalized = value.lower().replace("-", "")
    if normalized not in {"da20k", "xes3g5m"}:
        raise argparse.ArgumentTypeError("dataset must be da20k (DA-20K) or xes3g5m (XES3G5M)")
    return normalized


def parser():
    root = argparse.ArgumentParser(description="知识点预测评测：DA-20K / XES3G5M")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="转换数据、划分题目、生成扰动输入")
    prepare.add_argument("--dataset", required=True, type=dataset_name)
    prepare.add_argument("--questions", required=True, help="DA questions JSON/JSONL or XES metadata/questions.json")
    prepare.add_argument("--labels", required=True, help="DA knowledge JSON or XES metadata/kc_routes_map.json")
    prepare.add_argument("--links", help="DA qid/label_id table, unnecessary if questions contain labels")
    prepare.add_argument("--route-separator", help="XES route separator override")
    prepare.add_argument("--splits", help="Optional JSON with train/valid/test qid arrays")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--drop-prob", type=float, default=0.35)
    prepare.add_argument("--swap-prob", type=float, default=0.35)
    prepare.add_argument("--output-dir", required=True)
    embeddings = commands.add_parser("build-embeddings", help="生成可离线使用的知识点语义向量")
    embeddings.add_argument("--taxonomy", required=True)
    embeddings.add_argument("--output", required=True)
    embeddings.add_argument("--model", default=DEFAULT_MODEL)
    embeddings.add_argument("--revision", help="Optional Hugging Face model commit SHA")
    embeddings.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate", help="读取预测文件，输出评测报告")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--noisy-predictions", help="Predictions on test.noisy.inputs.jsonl")
    evaluate.add_argument("--embeddings", help="Defaults to DATA_DIR/embeddings.json")
    evaluate.add_argument("--skip-semantic", action="store_true", help="Explicit partial evaluation without semantic metrics")
    evaluate.add_argument("--allow-empty-predictions", action="store_true", help="Score abstentions as empty sets; marks submission noncompliant")
    evaluate.add_argument("--output", required=True, help="Summary JSON, including per-label scores")
    evaluate.add_argument("--details", help="Optional per-question JSONL output")
    evaluate.add_argument("--require-complete", action="store_true", help="Exit 1 when semantic/noisy predictions are missing")
    evaluate.add_argument("--fail-on-threshold", action="store_true", help="Exit 1 if strict<0.90 or loose<0.95")
    return root


def protect_outputs(outputs, inputs):
    resolved = [Path(path).resolve() for path in outputs if path]
    protected = {Path(path).resolve() for path in inputs if path}
    if len(set(resolved)) != len(resolved) or protected & set(resolved):
        raise ValueError("Output paths must be distinct and must not overwrite input files")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            if args.dataset == "da20k":
                if args.route_separator:
                    raise ValueError("--route-separator is only supported for XES3G5M")
                examples, taxonomy, info = load_da20k(args.questions, args.labels, args.links)
            else:
                if args.links:
                    raise ValueError("XES3G5M reads kc_routes from questions; do not supply --links")
                examples, taxonomy, info = load_xes3g5m(args.questions, args.labels, args.route_separator)
            result = prepare_dataset(examples, taxonomy, args.output_dir, args.seed, args.splits,
                                     args.drop_prob, args.swap_prob, info)
            print(json.dumps({"dataset": result["dataset"], "counts": result["counts"], "output_dir": args.output_dir}, ensure_ascii=False))
        elif args.command == "build-embeddings":
            protect_outputs([args.output], [args.taxonomy])
            result = build_embeddings(Taxonomy(read_json(args.taxonomy)), args.output, args.model, args.revision, args.device)
            print(f"Saved {len(result['vectors'])} embeddings to {args.output}")
        else:
            root = Path(args.data_dir)
            bundle_files = list(root.glob("*"))
            protect_outputs([args.output, args.details], [args.predictions, args.noisy_predictions, args.embeddings, *bundle_files])
            report, details = evaluate_submission(root, args.predictions, args.noisy_predictions, args.embeddings,
                                                 args.skip_semantic, args.allow_empty_predictions)
            write_json(args.output, report)
            if args.details:
                write_jsonl(args.details, details)
            metrics = report["metrics"]
            summary = {"dataset": report["dataset"], "strict_accuracy": metrics["strict_accuracy"],
                       "loose_accuracy": metrics["loose_accuracy"], "precision": metrics["exact"]["precision"],
                       "recall": metrics["exact"]["recall"], "micro_f1": metrics["exact"]["micro_f1"],
                       "macro_f1": metrics["exact"]["macro_f1"], "hierarchical": metrics["hierarchical"],
                       "semantic": metrics["semantic"], "interaction_robustness": report["interaction_robustness"]["ratio"],
                       "evaluation_complete": report["evaluation_complete"],
                       "accuracy_thresholds_passed": report["requirements"]["accuracy_thresholds_passed"], "report": args.output}
            print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
            if args.require_complete and not report["evaluation_complete"]:
                return 1
            if args.fail_on_threshold and not report["requirements"]["accuracy_thresholds_passed"]:
                return 1
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
