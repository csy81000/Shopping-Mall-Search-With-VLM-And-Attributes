"""Build OpenCLIP image/text embeddings, a FAISS index, and k-means clusters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _batches(values: list[Path], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero-length embedding")
    return matrix / norms


def fuse_embeddings(
    image_features: np.ndarray,
    text_features: np.ndarray,
    has_text: list[bool] | np.ndarray,
) -> np.ndarray:
    """Average normalized image/text rows and return unit-length embeddings."""

    image_features = _normalize_rows(image_features)
    text_features = _normalize_rows(text_features)
    if image_features.shape != text_features.shape:
        raise ValueError("Image and text embeddings must have the same shape")
    mask = np.asarray(has_text, dtype=bool)
    if mask.shape != (image_features.shape[0],):
        raise ValueError("has_text must contain one value per embedding row")
    combined = np.where(
        mask[:, None],
        (image_features + text_features) / 2,
        image_features,
    )
    return _normalize_rows(combined)


def build_index(
    image_root: Path,
    output_dir: Path,
    fusion: str = "image-text",
    clusters: int = 13,
    batch_size: int = 16,
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2b_s32b_b79k",
) -> None:
    """Encode a catalog and write portable index artifacts."""

    import faiss
    import open_clip
    import torch
    from PIL import Image
    from sklearn.cluster import KMeans

    if fusion not in {"image", "image-text"}:
        raise ValueError("fusion must be 'image' or 'image-text'")
    images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No catalog images found under {image_root}")
    if clusters < 1 or clusters > len(images):
        raise ValueError("clusters must be between 1 and the number of images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _train_transform, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    features: list[np.ndarray] = []
    descriptions: dict[str, str] = {}

    for batch in _batches(images, batch_size):
        image_tensors = []
        for path in batch:
            with Image.open(path) as image:
                image_tensors.append(preprocess(image.convert("RGB")))
        tensors = torch.stack(image_tensors).to(device)
        texts = []
        for path in batch:
            text_path = path.with_suffix(".txt")
            text = text_path.read_text(encoding="utf-8-sig").strip() if text_path.exists() else ""
            texts.append(text)
            descriptions[path.relative_to(image_root).as_posix()] = text
        with torch.inference_mode():
            image_features = model.encode_image(tensors)
            image_matrix = image_features.cpu().numpy()
            combined = _normalize_rows(image_matrix)
            if fusion == "image-text":
                nonempty = [text if text else "product" for text in texts]
                text_features = model.encode_text(tokenizer(nonempty).to(device))
                combined = fuse_embeddings(
                    image_matrix,
                    text_features.cpu().numpy(),
                    [bool(text) for text in texts],
                )
        features.append(combined.astype("float32"))

    matrix = np.concatenate(features, axis=0)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(matrix)
    relative_paths = [path.relative_to(image_root).as_posix() for path in images]

    cluster_data: dict[str, dict] = {}
    for cluster_id in range(clusters):
        members = [path for path, label in zip(relative_paths, labels) if label == cluster_id]
        representative_text = [descriptions[path] for path in members if descriptions[path]][:5]
        cluster_data[str(cluster_id)] = {
            "paths": members,
            "description": "\n".join(representative_text)[:3000]
            or f"Visual product cluster {cluster_id}",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "catalog.faiss"))
    (output_dir / "paths.json").write_text(
        json.dumps(relative_paths, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "clusters.json").write_text(
        json.dumps(cluster_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "fusion": fusion,
                "clusters": clusters,
                "items": len(images),
                "dimension": int(matrix.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
