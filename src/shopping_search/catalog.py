"""Portable FAISS catalog index loading and search."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class SearchResult:
    relative_path: str
    score: float
    cluster_id: int


class CatalogIndex:
    """FAISS index plus portable product-path and cluster metadata."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.index = faiss.read_index(str(directory / "catalog.faiss"))
        self.paths: list[str] = json.loads((directory / "paths.json").read_text(encoding="utf-8"))
        self.clusters: dict[str, dict] = json.loads(
            (directory / "clusters.json").read_text(encoding="utf-8")
        )
        self.metadata: dict = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
        self.path_to_cluster = {
            path: int(cluster_id)
            for cluster_id, value in self.clusters.items()
            for path in value["paths"]
        }
        if self.index.ntotal != len(self.paths):
            raise ValueError("FAISS row count and paths.json length do not match")

    def search(self, query: np.ndarray, candidate_k: int = 100, display_k: int = 10):
        query = np.asarray(query, dtype="float32")
        if query.ndim == 1:
            query = query[None, :]
        faiss.normalize_L2(query)
        candidate_k = min(candidate_k, self.index.ntotal)
        scores, indices = self.index.search(query, candidate_k)
        results = [
            SearchResult(
                relative_path=self.paths[index],
                score=float(score),
                cluster_id=self.path_to_cluster.get(self.paths[index], -1),
            )
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]
        cluster_ids = [result.cluster_id for result in results if result.cluster_id >= 0]
        major_cluster = Counter(cluster_ids).most_common(1)[0][0] if cluster_ids else -1
        return results[:display_k], major_cluster

    def cluster_description(self, cluster_id: int) -> str:
        if cluster_id < 0:
            return "No dominant product cluster was identified."
        value = self.clusters.get(str(cluster_id), {})
        return str(value.get("description") or f"Cluster {cluster_id}")
