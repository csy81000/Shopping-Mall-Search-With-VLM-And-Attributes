"""OpenCLIP query encoding."""

from __future__ import annotations

import numpy as np
import torch


class QueryEncoder:
    def __init__(self, model_name: str, pretrained: str) -> None:
        import open_clip

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _train, _preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode(self, text: str) -> np.ndarray:
        templates = [
            f"a product photo of {text}",
            f"a clear image showing {text}",
            f"a close-up product photo of {text}",
            f"{text} on a plain background",
            f"a shopping item described as {text}",
        ]
        with torch.inference_mode():
            tokens = self.tokenizer(templates).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.mean(dim=0, keepdim=True)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")

