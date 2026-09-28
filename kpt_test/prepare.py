"""Teacher-side preparation: stable question splits and noisy test inputs."""

import hashlib
import random
from pathlib import Path

from .io import index_records, write_json, write_jsonl, fingerprint, file_hash, read_json, identifier


def perturb_text(text, rng, drop_prob=0.35, swap_prob=0.35):
    """SGPE protocol: per-character deletion followed by adjacent swaps."""
    if not 0 <= drop_prob <= 1 or not 0 <= swap_prob <= 1:
        raise ValueError("Perturbation probabilities must be in [0, 1]")
    if len(text) <= 4:
        return text
    chars = list(text)
    kept = [ch for ch in chars if ch.isspace() or rng.random() > drop_prob]
    if len(kept) < max(1, len(chars) // 3):
        kept = chars[:]
    i = 0
    while i < len(kept) - 1:
        if not kept[i].isspace() and not kept[i + 1].isspace() and rng.random() < swap_prob:
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
            i += 2
        else:
            i += 1
    return "".join(kept)


def split_examples(examples, seed, split_file=None):
    indexed = index_records(examples)
    if split_file:
        supplied = read_json(split_file)
        if not isinstance(supplied, dict) or set(supplied) != {"train", "valid", "test"}:
            raise ValueError("Splits JSON must contain train, valid and test ID arrays")
        qids = {}
        seen = set()
        for name, values in supplied.items():
            if not isinstance(values, list):
                raise ValueError(f"Split {name} must be an ID array")
            ids = [identifier(value) for value in values]
            if len(set(ids)) != len(ids) or seen & set(ids):
                raise ValueError("Duplicate qid within/across splits")
            seen.update(ids)
            qids[name] = sorted(ids)
        if seen != set(indexed):
            raise ValueError("Splits must cover exactly the supplied question IDs")
    else:
        ordered = sorted(indexed, key=lambda q: (hashlib.sha256(f"{seed}:{q}".encode()).digest(), q))
        n = len(ordered)
        if n < 10:
            raise ValueError("Automatic 8:1:1 split needs at least 10 questions; use --splits for a small example")
        train_end, valid_end = int(n * 0.8), int(n * 0.9)
        qids = {"train": ordered[:train_end], "valid": ordered[train_end:valid_end], "test": ordered[valid_end:]}
    if not qids["test"]:
        raise ValueError("Test split must not be empty")
    return {name: [indexed[qid] for qid in sorted(ids)] for name, ids in qids.items()}


def prepare_dataset(examples, taxonomy, output, seed=42, split_file=None, drop_prob=0.35, swap_prob=0.35, adapter_info=None):
    if not 0 <= drop_prob <= 1 or not 0 <= swap_prob <= 1:
        raise ValueError("Perturbation probabilities must be in [0, 1]")
    splits = split_examples(examples, seed, split_file)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new directory")
    noisy = []
    for row in splits["test"]:
        noise_seed = int(hashlib.sha256(f"{seed}:{row['qid']}".encode()).hexdigest(), 16)
        noisy.append({"qid": row["qid"], "text": perturb_text(row["text"], random.Random(noise_seed), drop_prob, swap_prob)})
    files = {}
    for name in ("train", "valid"):
        filename = f"{name}.jsonl"
        write_jsonl(output / filename, splits[name])
        files[filename] = file_hash(output / filename)
    payloads = {
        "test.inputs.jsonl": [{"qid": row["qid"], "text": row["text"]} for row in splits["test"]],
        "test.gold.jsonl": [{"qid": row["qid"], "labels": row["labels"]} for row in splits["test"]],
        "test.noisy.inputs.jsonl": noisy,
        "predictions.template.jsonl": [{"qid": row["qid"], "labels": []} for row in splits["test"]],
    }
    for filename, rows in payloads.items():
        write_jsonl(output / filename, rows)
        files[filename] = file_hash(output / filename)
    write_json(output / "taxonomy.json", taxonomy.data)
    files["taxonomy.json"] = file_hash(output / "taxonomy.json")
    split_ids = {name: [row["qid"] for row in rows] for name, rows in splits.items()}
    write_json(output / "splits.json", split_ids)
    files["splits.json"] = file_hash(output / "splits.json")
    manifest = {"schema_version": 1, "dataset": taxonomy.dataset, "seed": seed,
                "split_method": "provided" if split_file else "sha256(seed:qid), 8:1:1",
                "counts": {name: len(rows) for name, rows in splits.items()},
                "taxonomy_sha256": taxonomy.digest,
                "source_sha256": fingerprint(sorted(examples, key=lambda row: row["qid"])),
                "noise": {"protocol": "sgpe_character_delete_then_adjacent_swap_v1", "seed": seed,
                          "drop_probability": drop_prob, "swap_probability": swap_prob,
                          "changed_questions": sum(a["text"] != b["text"] for a, b in zip(splits["test"], noisy))},
                "adapter": adapter_info or {}, "files": files}
    write_json(output / "manifest.json", manifest)
    return manifest
