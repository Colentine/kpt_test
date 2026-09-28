"""Rebuild the checked-in text-only datasets from the checked-in sources."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import zipfile

from .adapters import da_taxonomy, load_xes3g5m
from .io import file_hash, identifier, index_records, write_json
from .prepare import prepare_dataset
from .text import normalize_question_text


def text_statistics(examples):
    groups = defaultdict(list)
    for row in examples:
        groups[row["text"]].append(tuple(row["labels"]))
    return {"duplicate_text_rows": sum(len(labels) - 1 for labels in groups.values()),
            "duplicate_text_groups": sum(len(labels) > 1 for labels in groups.values()),
            "ambiguous_text_groups": sum(len(set(labels)) > 1 for labels in groups.values())}


def build_da20k(archive, labels_file, output):
    taxonomy = da_taxonomy(labels_file)
    with zipfile.ZipFile(archive) as source:
        def table(filename):
            matches = [name for name in source.namelist() if name.endswith("/" + filename) or name == filename]
            if len(matches) != 1:
                raise ValueError(f"Archive must contain exactly one {filename}")
            return json.loads(source.read(matches[0]).decode("utf-8-sig"))
        questions = table("math_questions_content.json")
        links = table("math_questions_knowledgetag.json")
        archive_commit = source.comment.decode("ascii", errors="replace")
    index_records([{"qid": row["id"]} for row in questions])
    tags = defaultdict(set)
    for row in links:
        tags[identifier(row["qid"])].add(identifier(row["label_id"]))
    if set(tags) - {identifier(row["id"]) for row in questions}:
        raise ValueError("Source tags reference unknown questions")
    examples, exclusions, filtered_tags, image_qids = [], [], [], []
    for row in sorted(questions, key=lambda r: identifier(r["id"])):
        qid = identifier(row["id"])
        selected = sorted(tags[qid] & taxonomy.labels.keys())
        if not selected:
            exclusions.append({"qid": qid, "reason": "no_tags" if not tags[qid] else "no_target_labels",
                               "original_labels": sorted(tags[qid])})
            continue
        text = normalize_question_text(row["text"])
        if not text or re.fullmatch(r"\d+[、.．]?", text):
            exclusions.append({"qid": qid, "reason": "no_text", "original_labels": sorted(tags[qid])})
            continue
        ignored = sorted(tags[qid] - taxonomy.labels.keys())
        if ignored:
            filtered_tags.append({"qid": qid, "ignored_labels": ignored, "kept_labels": selected})
        if re.search(r"<img\b", row["text"], re.I):
            image_qids.append(qid)
        examples.append({"qid": qid, "text": text, "labels": selected})
    info = {**text_statistics(examples), "source_questions": len(questions), "included_questions": len(examples),
            "excluded_questions": len(exclusions), "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
            "target_vocabulary": "official_DA-20k_427_labels_with_father",
            "questions_with_out_of_vocabulary_tags_removed": len(filtered_tags),
            "questions_with_images_removed": len(image_qids),
            "text_preprocessing": "HTML_to_text_MathML_to_LaTeX_remove_images_and_footer_v1"}
    manifest = prepare_dataset(examples, taxonomy, output, adapter_info=info)
    write_json(Path(output) / "provenance.json", {
        **info, "source_archive": Path(archive).name, "archive_sha256": file_hash(archive),
        "source_archive_commit": archive_commit, "label_file_sha256": file_hash(labels_file),
        "excluded": exclusions, "filtered_tags": filtered_tags, "image_reference_qids": image_qids})
    return manifest


def build_xes3g5m(questions_file, labels_file, output):
    examples, taxonomy, info = load_xes3g5m(questions_file, labels_file)
    image_qids = []
    for row in examples:
        if re.search(r"\bquestion_\d+-image_\d+\b|<img\b", row["text"]):
            image_qids.append(row["qid"])
        row["text"] = normalize_question_text(row["text"])
        if not row["text"]:
            raise ValueError(f"XES question {row['qid']} has no text after image removal")
    info = {**info, **text_statistics(examples), "source_questions": len(examples), "included_questions": len(examples),
            "excluded_questions": 0, "questions_with_images_removed": len(image_qids),
            "text_preprocessing": "remove_image_references_and_HTML_v1"}
    manifest = prepare_dataset(examples, taxonomy, output, adapter_info=info)
    write_json(Path(output) / "provenance.json", {
        **info, "questions_sha256": file_hash(questions_file), "label_file_sha256": file_hash(labels_file),
        "image_reference_qids": image_qids})
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Rebuild bundled text-only DA-20K / XES3G5M data")
    parser.add_argument("--dataset", required=True, choices=["da20k", "xes3g5m"])
    parser.add_argument("--source-dir", default="datasets/sources")
    parser.add_argument("--output-dir", required=True, help="Must be empty")
    args = parser.parse_args()
    root = Path(args.source_dir) / args.dataset
    if args.dataset == "da20k":
        manifest = build_da20k(root / "mathdata-main.zip", root / "DA-20k-labels-with-father.json", args.output_dir)
    else:
        manifest = build_xes3g5m(root / "questions.json", root / "kc_routes_map.json", args.output_dir)
    print(json.dumps({"dataset": manifest["dataset"], "counts": manifest["counts"], "adapter": manifest["adapter"]}))


if __name__ == "__main__":
    main()
