#!/usr/bin/env python3
"""Run the released GeoVLMass checkpoint on packaged Amazon-1435 features."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from model import MassModel


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ("amazon_test_6771", "amazon_test_6698", "amazon_test_6829", "amazon_test_7521")


def load_checkpoint(path):
    """Support PyTorch releases before the weights_only keyword."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def metrics(predictions, targets):
    p = np.maximum(predictions.astype(np.float64), 1e-8)
    y = np.maximum(targets.astype(np.float64), 1e-8)
    log_p, log_y = np.log(p), np.log(y)
    correlation = np.corrcoef(log_p, log_y)[0, 1] if len(p) > 1 else 0.0
    return {
        "count": len(p),
        "mae_kg": float(np.mean(np.abs(predictions - targets))),
        "alde": float(np.mean(np.abs(log_p - log_y))),
        "ape": float(np.mean(np.abs(predictions - targets) / y)),
        "mnre": float(np.mean(np.minimum(p / y, y / p))),
        "r2_ls": float(correlation**2) if np.isfinite(correlation) else 0.0,
        "q": float(np.mean((p / y >= 0.5) & (p / y <= 2.0))),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate seed-0 Amazon mass estimates (kg).")
    parser.add_argument("--all", action="store_true", help="Evaluate all 1435 test samples.")
    parser.add_argument("--samples", nargs="+", metavar="ID", help="Evaluate selected sample IDs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-csv", type=Path, help="Write per-sample results to this path.")
    args = parser.parse_args()
    if args.all and args.samples:
        parser.error("--all and --samples are mutually exclusive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    data_path = ROOT / "data" / "amazon_1435_features.npz"
    checkpoint_path = ROOT / "checkpoints" / "seed0_best.pt"
    if not data_path.is_file() or not checkpoint_path.is_file():
        parser.error("release data or checkpoint missing; see README.md")

    with np.load(data_path, allow_pickle=False) as data:
        ids = data["sample_ids"].tolist()
        if len(ids) != 1435 or len(set(ids)) != 1435:
            raise ValueError("expected 1435 unique test sample IDs")
        index = {sample_id: i for i, sample_id in enumerate(ids)}
        requested = ids if args.all else list(args.samples or DEFAULT_CASES)
        missing = [sample_id for sample_id in requested if sample_id not in index]
        if missing:
            parser.error(f"unknown sample ID(s): {', '.join(missing)}")
        selected = np.asarray([index[sample_id] for sample_id in requested], dtype=np.int64)
        targets = data["target_kg"][selected].astype(np.float64)
        if np.any(targets <= 0):
            raise ValueError("target weights must be positive")
        fields = ("image_features", "global_text_feature", "component_text_feature", "geometry", "view_metadata")
        arrays = [data[field][selected] for field in fields]

    checkpoint = load_checkpoint(checkpoint_path)
    saved_args = checkpoint.get("args", {})
    if (saved_args.get("release") != "GeoVLMass" or saved_args.get("dataset") != "Amazon-1435"
            or saved_args.get("seed") != 0 or saved_args.get("feature_backend") != "clip_cache"):
        raise ValueError("checkpoint is not the expected seed-0 CLIP-cache model")
    model = MassModel(feature_dim=arrays[0].shape[-1]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            inputs = [torch.from_numpy(array[start:start + args.batch_size]).to(device) for array in arrays]
            predictions.extend(model(*inputs).cpu().tolist())
    predictions = np.asarray(predictions, dtype=np.float64)
    rows = []
    if not args.all:
        print("sample_id            predicted_kg    actual_kg    abs_error_kg    relative_error")
    for sample_id, prediction, target in zip(requested, predictions, targets):
        row = {
            "sample_id": sample_id,
            "prediction_kg": prediction,
            "target_kg": target,
            "absolute_error_kg": abs(prediction - target),
            "relative_error": abs(prediction - target) / target,
        }
        rows.append(row)
        if not args.all:
            print(f"{sample_id:<20} {prediction:>12.4f} {target:>12.4f} {row['absolute_error_kg']:>15.4f} {row['relative_error']:>16.2%}")
    if args.all:
        print(f"Evaluated {len(rows)} test samples")
    print(json.dumps(metrics(predictions, targets), indent=2))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
