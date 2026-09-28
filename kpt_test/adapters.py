"""Adapters for DA-20K question/tag tables and XES3G5M metadata."""

import json

from .io import identifier, read_json, read_records, index_records, id_list
from .taxonomy import Taxonomy


def question_rows(path):
    if str(path).lower().endswith(".jsonl"):
        return read_records(path)
    data = read_json(path)
    if isinstance(data, dict):
        rows = []
        for qid, value in data.items():
            if not isinstance(value, dict):
                raise ValueError("Question map values must be objects")
            if "qid" in value and identifier(value["qid"]) != qid:
                raise ValueError(f"Conflicting question ID: {qid}")
            rows.append({**value, "qid": qid})
        return rows
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("Questions must be an object keyed by qid or an array of objects")
    return data


def question_text(row):
    # Never include answer/analysis/knowledge paths as model input.
    text = next((row[key] for key in ("text_processed", "text", "content", "question") if row.get(key)), None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Question has no nonempty text/content/question/text_processed string")
    options = row.get("options")
    if options:
        if isinstance(options, dict):
            option_text = "\n".join(f"{key}: {value}" for key, value in options.items())
        elif isinstance(options, list):
            option_text = "\n".join(str(item) for item in options)
        elif isinstance(options, str):
            option_text = options
        else:
            raise ValueError("Question options must be null, a string, array, or object")
        text = text.rstrip() + "\n" + option_text
    return text.strip()


def da_taxonomy(path):
    data = read_json(path)
    if isinstance(data, dict) and "labels" in data:
        taxonomy = Taxonomy(data)
        if taxonomy.dataset != "da20k":
            raise ValueError("DA-20K requires a da20k taxonomy")
        return taxonomy
    if not isinstance(data, list) or not data:
        raise ValueError("DA-20K knowledge file must be a nonempty array")
    if all("label_father_id" in row for row in data):
        return Taxonomy({"dataset": "da20k", "labels": [
            {"id": identifier(row["label_id"]), "name": row["name"],
             "definition": row.get("definition", ""),
             "path": [identifier(row["label_father_id"]), identifier(row["label_id"])],
             "path_names": [row["label_father_name"], row["name"]]} for row in data]})
    by_id, uuid_to_id = {}, {}
    for row in data:
        kid = identifier(row.get("id", row.get("label_id")))
        if kid in by_id:
            raise ValueError(f"Duplicate knowledge ID: {kid}")
        by_id[kid] = row
        if row.get("uuid"):
            uuid = identifier(row["uuid"])
            if uuid in uuid_to_id:
                raise ValueError(f"Duplicate knowledge UUID: {uuid}")
            uuid_to_id[uuid] = kid
    parents = {}
    for kid, row in by_id.items():
        parent_uuid = row.get("parent_uuid")
        if parent_uuid:
            if str(parent_uuid) not in uuid_to_id:
                raise ValueError(f"Knowledge {kid}: missing parent UUID {parent_uuid}")
            parents[kid] = uuid_to_id[str(parent_uuid)]
        elif row.get("parent_id") is not None:
            parents[kid] = identifier(row["parent_id"])
            if parents[kid] not in by_id:
                raise ValueError(f"Knowledge {kid}: missing parent ID")
        else:
            parents[kid] = None
    paths = {}
    for kid in by_id:
        path_ids, current = [], kid
        while current is not None:
            if current in path_ids:
                raise ValueError(f"Knowledge hierarchy contains a cycle at {current}")
            path_ids.append(current)
            current = parents[current]
        paths[kid] = list(reversed(path_ids))
    internal = set(parents.values()) - {None}
    return Taxonomy({"dataset": "da20k", "labels": [
        {"id": kid, "name": row["name"], "definition": row.get("definition", ""),
         "path": paths[kid], "path_names": [by_id[node]["name"] for node in paths[kid]]}
        for kid, row in by_id.items() if kid not in internal]})


def load_da20k(questions_path, labels_path, links_path=None):
    taxonomy = da_taxonomy(labels_path)
    tags = {}
    if links_path:
        for row in read_records(links_path):
            qid, kid = identifier(row.get("qid")), identifier(row.get("label_id"))
            tags.setdefault(qid, set()).add(kid)
    examples, removed_ancestors = [], 0
    for row in question_rows(questions_path):
        qid = identifier(row.get("qid", row.get("id")))
        if links_path:
            labels = sorted(tags.get(qid, set()))
        else:
            labels = id_list(row.get("labels", row.get("label_ids")), f"qid={qid}")
        if not labels:
            raise ValueError(f"Question {qid}: no labels; supply the corresponding --links file")
        leaves = [kid for kid in labels if kid in taxonomy.labels]
        ancestors = {node for kid in leaves for node in taxonomy.labels[kid]["path"][:-1]}
        invalid = set(labels) - set(leaves) - ancestors
        if invalid or not leaves:
            raise ValueError(f"Question {qid}: unknown/non-leaf tags without a tagged descendant: {sorted(invalid or set(labels))}")
        removed_ancestors += len(set(labels) & ancestors)
        examples.append({"qid": qid, "text": question_text(row), "labels": sorted(leaves)})
    indexed = index_records(examples)
    if set(tags) - indexed.keys():
        raise ValueError("Tag table contains qids absent from question table")
    return examples, taxonomy, {"removed_redundant_ancestor_tags": removed_ancestors}


def parse_route(value, separator=None):
    if isinstance(value, list):
        parts = value
    elif isinstance(value, str):
        # XES releases use textual routes. Override for a different export.
        delimiter = separator or next((s for s in ("----", "->", ">", "/", "::") if s in value), None)
        parts = value.split(delimiter) if delimiter else [value]
    else:
        raise ValueError(f"Knowledge route must be a string or an array of names: {value!r}")
    if not parts or any(not isinstance(part, str) or not part.strip() for part in parts):
        raise ValueError(f"Invalid knowledge route: {value!r}")
    return tuple(part.strip() for part in parts)


def load_xes3g5m(questions_path, labels_path, separator=None):
    routes = read_json(labels_path)
    if not isinstance(routes, dict) or not routes:
        raise ValueError("kc_routes_map.json must map label IDs to textual routes")
    # The official release maps IDs to individual node names (all 1,175 nodes),
    # while questions contain full routes. Exports may instead map IDs to paths.
    questions = question_rows(questions_path)
    delimiter = separator or "----"
    if all(isinstance(value, str) and delimiter not in value for value in routes.values()) and any(
            isinstance(route, str) and delimiter in route
            for row in questions for route in row.get("kc_routes", [])):
        return load_xes_node_map(questions, routes, delimiter)
    route_to_id, names_to_ids, labels = {}, {}, []
    for raw_id, value in routes.items():
        kid = identifier(raw_id)
        route = parse_route(value, separator)
        if route in route_to_id:
            raise ValueError(f"Duplicate XES route: {route}")
        route_to_id[route] = kid
        names_to_ids.setdefault(route[-1], []).append(kid)
        # Prefix IDs encode the complete ancestry, so repeated node names do not collide.
        path = ["xes-path:" + json.dumps(route[:i], ensure_ascii=False, separators=(",", ":")) for i in range(1, len(route))]
        labels.append({"id": kid, "name": route[-1], "path": path + [kid], "path_names": list(route)})
    # Some exports include internal KCs in their map. Only terminal routes are targets.
    internal_routes = {route[:i] for route in route_to_id for i in range(1, len(route))}
    internal_ids = {kid for route, kid in route_to_id.items() if route in internal_routes}
    taxonomy = Taxonomy({"dataset": "xes3g5m", "labels": [row for row in labels if row["id"] not in internal_ids]})
    examples, removed_ancestors = [], 0
    for row in questions:
        qid = identifier(row.get("qid", row.get("id")))
        raw_routes = row.get("kc_routes")
        if not isinstance(raw_routes, list) or not raw_routes:
            raise ValueError(f"Question {qid}: kc_routes must be a nonempty array of routes")
        selected_routes = []
        for raw_route in raw_routes:
            route = parse_route(raw_route, separator)
            kid = route_to_id.get(route)
            if kid is None and len(route) == 1 and len(names_to_ids.get(route[0], [])) == 1:
                kid = names_to_ids[route[0]][0]
                route = next(r for r, k in route_to_id.items() if k == kid)
            if kid is None:
                raise ValueError(f"Question {qid}: unknown or ambiguous route {raw_route!r}")
            selected_routes.append((route, kid))
        leaves = sorted({kid for _, kid in selected_routes if kid in taxonomy.labels})
        for route, kid in selected_routes:
            if kid in internal_ids:
                if not any(other[:len(route)] == route and len(other) > len(route) and other_id in leaves for other, other_id in selected_routes):
                    raise ValueError(f"Question {qid}: internal KC {kid} has no tagged leaf descendant")
                removed_ancestors += 1
        if not leaves:
            raise ValueError(f"Question {qid}: no leaf knowledge points")
        examples.append({"qid": qid, "text": question_text(row), "labels": leaves})
    index_records(examples)
    return examples, taxonomy, {"removed_redundant_ancestor_tags": removed_ancestors}


def load_xes_node_map(questions, node_map, separator=None):
    name_to_id = {}
    for raw_id, name in node_map.items():
        kid = identifier(raw_id)
        # Three official IDs differ only by leading/trailing spaces. Preserve
        # exact source names, otherwise the 865-label vocabulary changes.
        if name in name_to_id:
            raise ValueError(f"Ambiguous XES node name: {name}")
        name_to_id[name] = kid
    label_routes, examples = {}, []
    for row in questions:
        qid = identifier(row.get("qid", row.get("id")))
        raw_routes = row.get("kc_routes")
        if not isinstance(raw_routes, list) or not raw_routes:
            raise ValueError(f"Question {qid}: kc_routes must be a nonempty array")
        labels = set()
        for raw_route in raw_routes:
            if isinstance(raw_route, str):
                names = tuple(raw_route.split(separator or "----"))
            elif isinstance(raw_route, list):
                names = tuple(raw_route)
            else:
                raise ValueError(f"Question {qid}: invalid route")
            try:
                path = tuple(name_to_id[name] for name in names)
            except KeyError as exc:
                raise ValueError(f"Question {qid}: unmapped XES route node {exc.args[0]}") from exc
            if len(set(path)) != len(path):
                raise ValueError(f"Question {qid}: repeated node in route")
            labels.add(path[-1])
            label_routes.setdefault(path[-1], {})[path] = names
        examples.append({"qid": qid, "text": question_text(row), "labels": sorted(labels)})
    index_records(examples)
    taxonomy = Taxonomy({"dataset": "xes3g5m", "target_policy": "route_endpoints", "labels": [
        {"id": kid, "name": node_map[kid], "routes": [
            {"path": list(path), "path_names": list(names)} for path, names in paths.items()]}
        for kid, paths in label_routes.items()]})
    return examples, taxonomy, {"target_policy": "route_endpoints", "source_nodes": len(node_map),
                               "labels_with_multiple_routes": sum(len(paths) > 1 for paths in label_routes.values())}
