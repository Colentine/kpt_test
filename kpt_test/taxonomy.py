"""A taxonomy stores root-to-leaf node IDs separately from display names."""

from .io import identifier, id_list, fingerprint


class Taxonomy:
    def __init__(self, data):
        if not isinstance(data, dict) or data.get("dataset") not in {"da20k", "xes3g5m"}:
            raise ValueError("Taxonomy dataset must be da20k or xes3g5m")
        self.dataset = data["dataset"]
        self.route_endpoints = data.get("target_policy") == "route_endpoints"
        if self.route_endpoints and self.dataset != "xes3g5m":
            raise ValueError("route_endpoints is only supported for XES3G5M")
        rows = data.get("labels")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Taxonomy must contain a nonempty labels array")
        self.labels = {}
        node_names = {}
        node_parents = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Taxonomy label must be an object")
            kid = identifier(row.get("id"))
            if kid in self.labels:
                raise ValueError(f"Duplicate taxonomy label: {kid}")
            name = row.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Label {kid}: missing name")
            definition = row.get("definition", "")
            if not isinstance(definition, str):
                raise ValueError(f"Label {kid}: definition must be a string")
            routes = row.get("routes") if self.route_endpoints else None
            if routes is None:
                routes = [{"path": row.get("path"), "path_names": row.get("path_names")}]
            if not isinstance(routes, list) or not routes or not all(isinstance(r, dict) for r in routes):
                raise ValueError(f"Label {kid}: routes must be a nonempty array")
            checked = []
            for route in routes:
                path = id_list(route.get("path"), f"Label {kid} path")
                if path[-1] != kid:
                    raise ValueError(f"Label {kid}: path must end in its own ID")
                names = route.get("path_names")
                if not isinstance(names, list) or len(names) != len(path) or any(not isinstance(x, str) or not x.strip() for x in names):
                    raise ValueError(f"Label {kid}: path_names must align with path")
                if names[-1] != name:
                    raise ValueError(f"Label {kid}: last path name must equal name")
                for i, (node, node_name) in enumerate(zip(path, names)):
                    parent = path[i - 1] if i else None
                    if node in node_names and (node_names[node] != node_name or (not self.route_endpoints and node_parents[node] != parent)):
                        raise ValueError(f"Conflicting taxonomy node: {node}")
                    node_names[node], node_parents[node] = node_name, parent
                checked.append({"path": path, "path_names": names})
            checked.sort(key=lambda r: r["path"])
            if len({tuple(r["path"]) for r in checked}) != len(checked):
                raise ValueError(f"Label {kid}: duplicate routes")
            self.labels[kid] = {"id": kid, "name": name, "definition": definition, **checked[0]}
            if self.route_endpoints:
                self.labels[kid]["routes"] = checked
        internal = {node for row in self.labels.values() for node in row["path"][:-1]}
        if not self.route_endpoints and internal & self.labels.keys():
            raise ValueError(f"Non-leaf labels in vocabulary: {sorted(internal & self.labels.keys())[:5]}")
        self.ids = sorted(self.labels)
        self.data = {"schema_version": 1, "dataset": self.dataset, "labels": [self.labels[k] for k in self.ids]}
        if self.route_endpoints:
            self.data["target_policy"] = "route_endpoints"
        self.digest = fingerprint(self.data)

    def validate_labels(self, labels, context):
        invalid = set(labels) - self.labels.keys()
        if invalid:
            raise ValueError(f"{context}: unknown/non-leaf label IDs: {sorted(invalid)[:10]}")

    def hierarchy_score(self, pred, truth):
        if pred == truth:
            return 1.0
        return max(self._path_score(p["path"], t["path"])
                   for p in self.routes(pred) for t in self.routes(truth))

    @staticmethod
    def _path_score(p, t):
        if len(p) > 1 and len(t) > 1 and p[-2] == t[-2]:
            return 0.5
        if len(p) > 2 and len(t) > 2 and p[-3] == t[-3]:
            return 0.2
        return 0.05 if set(p[:-1]) & set(t[:-1]) else 0.0

    def text(self, kid):
        row = self.labels[kid]
        paths = " | ".join(" / ".join(r["path_names"]) for r in self.routes(kid))
        return " [SEP] ".join(part for part in (row["name"], row["definition"], paths) if part)

    def routes(self, kid):
        row = self.labels[kid]
        return row.get("routes", [row])
