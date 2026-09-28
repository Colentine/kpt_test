"""Exact and one-to-one optimal soft matching, with micro aggregation."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def prf(tp, pred_count, truth_count):
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / truth_count if truth_count else 0.0
    return {"tp": tp, "fp": max(0.0, pred_count - tp), "fn": max(0.0, truth_count - tp),
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (pred_count + truth_count) if pred_count + truth_count else 0.0}


def optimal_match(pred, truth, score):
    if not pred or not truth:
        return 0.0
    matrix = np.array([[score(p, t) for t in truth] for p in pred], dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Pair scores must be finite numbers in [0, 1]")
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    return float(matrix[rows, columns].sum())


def evaluate_sets(gold, predictions, taxonomy, semantic_score=None):
    """Inputs have already passed strict ID, uniqueness and vocabulary validation."""
    if not gold or set(gold) != set(predictions):
        raise ValueError("Gold and predictions must contain identical, nonempty qid sets")
    hits = loose = pred_count = truth_count = 0
    exact_tp = Counter()
    predicted = Counter()
    support = Counter()
    hierarchical_tp = semantic_tp = 0.0
    details = []
    for qid in sorted(gold):
        truth, pred = sorted(gold[qid]), sorted(predictions[qid])
        ts, ps = set(truth), set(pred)
        correct = ts & ps
        hits += ps == ts
        loose += bool(correct)
        exact_tp.update(correct)
        predicted.update(pred)
        support.update(truth)
        pred_count += len(pred)
        truth_count += len(truth)
        htp = optimal_match(pred, truth, taxonomy.hierarchy_score)
        stp = optimal_match(pred, truth, semantic_score) if semantic_score else None
        hierarchical_tp += htp
        if stp is not None:
            semantic_tp += stp
        details.append({"qid": qid, "gold_labels": truth, "predicted_labels": pred,
                        "strict_correct": ps == ts, "loose_correct": bool(correct),
                        "exact": prf(len(correct), len(pred), len(truth)),
                        "hierarchical": prf(htp, len(pred), len(truth)),
                        "semantic": prf(stp, len(pred), len(truth)) if stp is not None else None})
    per_label = [{"id": kid, "name": taxonomy.labels[kid]["name"], "support": support[kid],
                  **prf(exact_tp[kid], predicted[kid], support[kid])} for kid in taxonomy.ids]
    micro = prf(sum(exact_tp.values()), pred_count, truth_count)
    return {"num_questions": len(gold), "num_labels": len(taxonomy.ids),
            "strict_accuracy": hits / len(gold), "loose_accuracy": loose / len(gold),
            "exact": {**micro, "micro_f1": micro["f1"],
                      "macro_f1": sum(row["f1"] for row in per_label) / len(per_label),
                      "macro_precision": sum(row["precision"] for row in per_label) / len(per_label),
                      "macro_recall": sum(row["recall"] for row in per_label) / len(per_label),
                      "macro_scope": "all_taxonomy_leaf_labels", "zero_division": 0},
            "hierarchical": prf(hierarchical_tp, pred_count, truth_count),
            "semantic": prf(semantic_tp, pred_count, truth_count) if semantic_score else None,
            "per_label": per_label}, details
