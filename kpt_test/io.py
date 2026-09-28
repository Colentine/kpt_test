"""Strict JSON input and reproducible serialization."""

import hashlib
import json
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line, object_pairs_hook=unique_object))
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    else:
        rows = read_json(path)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected a JSON array of objects or JSONL")
    return rows


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"ID must be a string or integer, got {value!r}")
    result = str(value)
    if not result.strip() or result != result.strip():
        raise ValueError(f"Empty ID or surrounding whitespace: {value!r}")
    return result


def id_list(value, context, allow_empty=False):
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected an array of label IDs")
    result = [identifier(item) for item in value]
    if not result and not allow_empty:
        raise ValueError(f"{context}: at least one label is required")
    if len(result) != len(set(result)):
        raise ValueError(f"{context}: duplicate label IDs")
    return result


def index_records(rows, label_field=None, allow_empty=False):
    result = {}
    for row in rows:
        if "qid" not in row:
            raise ValueError("Every record must contain qid")
        qid = identifier(row["qid"])
        if qid in result:
            raise ValueError(f"Duplicate qid: {qid}")
        value = row
        if label_field:
            value = id_list(row.get(label_field), f"qid={qid}, {label_field}", allow_empty)
        result[qid] = value
    if not result:
        raise ValueError("Input contains no records")
    return result


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(obj):
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
