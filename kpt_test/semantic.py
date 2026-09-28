"""Generate once, then evaluate offline with a validated embedding cache."""

import numpy as np

from .io import read_json, write_json

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_embeddings(taxonomy, output, model_name=DEFAULT_MODEL, revision=None, device="cpu"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ValueError('Install semantic dependencies first: pip install ".[semantic]"') from exc
    model = SentenceTransformer(model_name, revision=revision, device=device)
    config = getattr(getattr(model[0], "auto_model", None), "config", None)
    resolved_revision = getattr(config, "_commit_hash", None)
    vectors = model.encode([taxonomy.text(kid) for kid in taxonomy.ids], batch_size=64,
                           show_progress_bar=True, normalize_embeddings=True)
    payload = {"schema_version": 1, "taxonomy_sha256": taxonomy.digest,
               "model": model_name, "revision": resolved_revision or revision,
               "text_format": "name [SEP] definition_if_present [SEP] root / ... / leaf",
               "vectors": {kid: vector.tolist() for kid, vector in zip(taxonomy.ids, vectors)}}
    # Validate generated vectors before writing a distributable cache.
    SemanticScores(payload, taxonomy)
    write_json(output, payload)
    return payload


class SemanticScores:
    def __init__(self, payload, taxonomy):
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported semantic cache schema")
        if payload.get("taxonomy_sha256") != taxonomy.digest:
            raise ValueError("Semantic cache taxonomy hash mismatch; rebuild embeddings")
        if not isinstance(payload.get("model"), str) or not payload["model"].strip():
            raise ValueError("Semantic cache must identify its model")
        vectors = payload.get("vectors")
        if not isinstance(vectors, dict) or set(vectors) != set(taxonomy.ids):
            raise ValueError("Semantic cache must contain exactly all taxonomy label IDs")
        if any(not isinstance(v, list) or not v or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v) for v in vectors.values()):
            raise ValueError("Embedding vectors must be nonempty numeric arrays")
        try:
            matrix = np.asarray([vectors[kid] for kid in taxonomy.ids], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Embedding dimensions must agree") from exc
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("Embeddings must be finite vectors with equal dimensions")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Embeddings must have finite, nonzero norms")
        normalized = matrix / norms[:, None]
        self.similarities = np.clip(normalized @ normalized.T, 0, 1)
        np.fill_diagonal(self.similarities, 1.0)
        self.index = {kid: i for i, kid in enumerate(taxonomy.ids)}
        self.metadata = {"model": payload["model"], "revision": payload.get("revision"),
                         "dimension": matrix.shape[1], "taxonomy_sha256": taxonomy.digest}

    def __call__(self, pred, truth):
        return float(self.similarities[self.index[pred], self.index[truth]])


def load_embeddings(path, taxonomy):
    return SemanticScores(read_json(path), taxonomy)
