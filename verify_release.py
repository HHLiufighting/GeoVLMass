#!/usr/bin/env python3
"""Check that a downloaded GeoVLMass release is complete and self-contained."""

from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / "requirements.txt",
    ROOT / "demo.py",
    ROOT / "model.py",
    ROOT / "checkpoints" / "seed0_best.pt",
    ROOT / "data" / "amazon_1435_features.npz",
    ROOT / "data" / "test_manifest.jsonl",
    ROOT / "assets" / "Figure1.pdf",
    ROOT / "assets" / "Figure1.png",
)
CASE_IDS = ("amazon_test_6771", "amazon_test_6698", "amazon_test_6829", "amazon_test_7521")


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit("Missing release files: " + ", ".join(missing))
    with np.load(ROOT / "data" / "amazon_1435_features.npz", allow_pickle=False) as data:
        expected = {
            "sample_ids": (1435,),
            "image_features": (1435, 5, 512),
            "global_text_feature": (1435, 512),
            "component_text_feature": (1435, 512),
            "geometry": (1435, 6),
            "view_metadata": (1435, 5, 3),
            "target_kg": (1435,),
        }
        for name, shape in expected.items():
            if name not in data or data[name].shape != shape:
                raise SystemExit(f"Unexpected {name} shape: {None if name not in data else data[name].shape}; expected {shape}")
        sample_ids = data["sample_ids"].tolist()
        if len(set(sample_ids)) != 1435 or any(not item.startswith("amazon_test_") for item in sample_ids):
            raise SystemExit("sample_ids are not a unique Amazon-1435 test split")
    checkpoint = load_checkpoint(ROOT / "checkpoints" / "seed0_best.pt")
    args = checkpoint.get("args", {})
    if (args.get("release") != "GeoVLMass" or args.get("dataset") != "Amazon-1435"
            or args.get("seed") != 0 or args.get("feature_backend") != "clip_cache"):
        raise SystemExit("checkpoint is not the expected seed-0 clip-cache model")
    if not isinstance(checkpoint.get("model"), dict) or not checkpoint["model"]:
        raise SystemExit("checkpoint has no model state")
    missing_cases = [case_id for case_id in CASE_IDS if not (ROOT / "examples" / case_id / "contact_sheet.png").is_file()]
    if missing_cases:
        raise SystemExit("Missing example cases: " + ", ".join(missing_cases))
    print("GeoVLMass release verification passed")
    print("  test samples: 1435")
    print("  checkpoint: seed 0")
    print("  pipeline: assets/Figure1.pdf and assets/Figure1.png")


if __name__ == "__main__":
    main()
