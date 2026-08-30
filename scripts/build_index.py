"""Build OpenCLIP/FAISS catalog artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from shopping_search.indexing import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fusion", choices=("image", "image-text"), default="image-text")
    parser.add_argument("--clusters", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        args.image_root,
        args.output_dir,
        args.fusion,
        args.clusters,
        args.batch_size,
        args.model_name,
        args.pretrained,
    )
    print(f"Index written to {args.output_dir}")


if __name__ == "__main__":
    main()

