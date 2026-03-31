"""
Time-series data preparation and evaluation utilities for autoresearch-ts-anomaly.

Usage:
    python prepare.py

The script creates a small synthetic multivariate dataset in ./data/ when no
dataset exists yet. Runtime helpers are imported by train.py.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_LEN = 128
PREDICTION_HORIZON = 16
TIME_BUDGET = 300
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DEFAULT_BATCH_SIZE = 64
DEFAULT_FEATURE_DIM = 4
NORMALIZATION_EPS = 1e-6

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "synthetic_timeseries.npz"


# ---------------------------------------------------------------------------
# Dataset containers
# ---------------------------------------------------------------------------

@dataclass
class PreparedSplit:
    values: torch.Tensor
    labels: torch.Tensor


@dataclass
class PreparedData:
    train: PreparedSplit
    val: PreparedSplit
    test: PreparedSplit
    mean: torch.Tensor
    std: torch.Tensor
    num_features: int
    context_len: int
    prediction_horizon: int


class TimeSeriesWindowDataset(torch.utils.data.Dataset):
    def __init__(self, values: torch.Tensor, labels: torch.Tensor, context_len: int, horizon: int):
        if values.ndim != 2:
            raise ValueError(f"Expected values with shape [time, features], got {tuple(values.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"Expected labels with shape [time], got {tuple(labels.shape)}")
        if values.size(0) != labels.size(0):
            raise ValueError("values and labels must have the same time dimension")
        if values.size(0) < context_len + horizon:
            raise ValueError("sequence is shorter than context_len + horizon")
        self.values = values
        self.labels = labels
        self.context_len = context_len
        self.horizon = horizon

    def __len__(self) -> int:
        return self.values.size(0) - self.context_len - self.horizon + 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx
        mid = start + self.context_len
        end = mid + self.horizon
        context = self.values[start:mid]
        future_values = self.values[mid:end]
        future_labels = self.labels[mid:end]
        anomaly_target = future_labels.max()
        return {
            "context": context,
            "future_values": future_values,
            "future_labels": future_labels,
            "anomaly_target": anomaly_target,
        }


# ---------------------------------------------------------------------------
# Data IO
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    output_path: Path = DEFAULT_DATASET_PATH,
    num_steps: int = 8000,
    num_features: int = DEFAULT_FEATURE_DIM,
    anomaly_rate: float = 0.02,
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps, dtype=np.float32)

    values = []
    for feature_idx in range(num_features):
        freq = 0.005 * (feature_idx + 1)
        seasonal = np.sin(2 * np.pi * freq * t)
        trend = 0.0005 * (feature_idx + 1) * t
        noise = 0.08 * rng.standard_normal(num_steps, dtype=np.float32)
        values.append(seasonal + trend + noise)
    values = np.stack(values, axis=1).astype(np.float32)

    labels = np.zeros(num_steps, dtype=np.float32)
    num_anomalies = max(1, int(num_steps * anomaly_rate))
    anomaly_positions = rng.choice(np.arange(CONTEXT_LEN, num_steps - PREDICTION_HORIZON), size=num_anomalies, replace=False)

    for pos in anomaly_positions:
        width = int(rng.integers(4, 16))
        amp = rng.uniform(2.0, 4.5)
        channel = int(rng.integers(0, num_features))
        end = min(num_steps, pos + width)
        values[pos:end, channel] += amp
        labels[pos:end] = 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, values=values, labels=labels)
    return output_path


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.load(path)
    values = data["values"].astype(np.float32)
    labels = data["labels"].astype(np.float32) if "labels" in data.files else None
    return values, labels


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    frame = pd.read_csv(path)
    label_col = next((col for col in frame.columns if col.lower() in {"label", "labels", "anomaly", "target"}), None)
    labels = frame.pop(label_col).to_numpy(dtype=np.float32) if label_col else None
    values = frame.select_dtypes(include=["number"]).to_numpy(dtype=np.float32)
    if values.size == 0:
        raise ValueError(f"No numeric feature columns found in {path}")
    return values, labels


def _weak_labels_from_diff(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    diffs = np.diff(values, axis=0, prepend=values[[0]])
    z = np.abs((diffs - diffs.mean(axis=0, keepdims=True)) / (diffs.std(axis=0, keepdims=True) + NORMALIZATION_EPS))
    return (z.max(axis=1) > threshold).astype(np.float32)


def load_raw_dataset(path: Path | str | None = None) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not dataset_path.exists():
        dataset_path = generate_synthetic_dataset(dataset_path)

    if dataset_path.suffix == ".npz":
        values, labels = _load_npz(dataset_path)
    elif dataset_path.suffix == ".csv":
        values, labels = _load_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    if values.ndim == 1:
        values = values[:, None]
    if labels is None:
        labels = _weak_labels_from_diff(values)
    labels = labels.reshape(-1).astype(np.float32)
    if len(values) != len(labels):
        raise ValueError("values and labels must have matching length")
    return values.astype(np.float32), labels


# ---------------------------------------------------------------------------
# Preparation and loaders
# ---------------------------------------------------------------------------

def _split_lengths(total_steps: int) -> tuple[int, int, int]:
    train_end = int(total_steps * TRAIN_RATIO)
    val_end = train_end + int(total_steps * VAL_RATIO)
    test_end = total_steps
    return train_end, val_end, test_end


def _normalize_splits(values: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[:train_end].mean(axis=0, keepdims=True)
    std = values[:train_end].std(axis=0, keepdims=True)
    normalized = (values - mean) / (std + NORMALIZATION_EPS)
    return normalized, mean.squeeze(0), std.squeeze(0)


def prepare_data(
    dataset_path: Path | str | None = None,
    context_len: int = CONTEXT_LEN,
    prediction_horizon: int = PREDICTION_HORIZON,
) -> PreparedData:
    values_np, labels_np = load_raw_dataset(dataset_path)
    train_end, val_end, test_end = _split_lengths(len(values_np))
    normalized_values, mean_np, std_np = _normalize_splits(values_np, train_end)

    values = torch.from_numpy(normalized_values)
    labels = torch.from_numpy(labels_np)

    train = PreparedSplit(values=values[:train_end], labels=labels[:train_end])
    val = PreparedSplit(values=values[train_end:val_end], labels=labels[train_end:val_end])
    test = PreparedSplit(values=values[val_end:test_end], labels=labels[val_end:test_end])

    minimum_len = context_len + prediction_horizon
    for split_name, split in {"train": train, "val": val, "test": test}.items():
        if split.values.size(0) < minimum_len:
            raise ValueError(f"{split_name} split is too short for context={context_len} horizon={prediction_horizon}")

    return PreparedData(
        train=train,
        val=val,
        test=test,
        mean=torch.from_numpy(mean_np.astype(np.float32)),
        std=torch.from_numpy(std_np.astype(np.float32)),
        num_features=values.size(1),
        context_len=context_len,
        prediction_horizon=prediction_horizon,
    )


def make_dataloader(
    split: PreparedSplit,
    batch_size: int,
    context_len: int,
    prediction_horizon: int,
    shuffle: bool,
    device: str | torch.device,
) -> torch.utils.data.DataLoader:
    dataset = TimeSeriesWindowDataset(split.values, split.labels, context_len, prediction_horizon)

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        collated = {}
        for key in batch[0]:
            stacked = torch.stack([item[key] for item in batch], dim=0)
            collated[key] = stacked
        return collated

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=(str(device).startswith("cuda")),
        collate_fn=collate,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def binary_pr_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    positives = labels.sum().item()
    if positives <= 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / positives
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    return torch.trapz(precision, recall).item()


def best_f1_threshold(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float, float]:
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    thresholds = torch.unique(scores)
    if thresholds.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0

    best_f1 = 0.0
    best_threshold = thresholds[0].item()
    best_precision = 0.0
    best_recall = 0.0
    for threshold in thresholds:
        preds = (scores >= threshold).float()
        tp = (preds * labels).sum().item()
        fp = (preds * (1.0 - labels)).sum().item()
        fn = ((1.0 - preds) * labels).sum().item()
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold.item()
            best_precision = precision
            best_recall = recall
    return best_f1, best_threshold, best_precision, best_recall


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, anomaly_loss_weight: float = 1.0) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_forecast_loss = 0.0
    total_anomaly_loss = 0.0
    batches = 0
    score_list = []
    label_list = []

    for batch in dataloader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        outputs = model(batch["context"])
        forecast_loss = torch.nn.functional.mse_loss(outputs["future_values"], batch["future_values"])
        anomaly_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["anomaly_logits"], batch["anomaly_target"]
        )
        loss = forecast_loss + anomaly_loss_weight * anomaly_loss
        total_loss += loss.item()
        total_forecast_loss += forecast_loss.item()
        total_anomaly_loss += anomaly_loss.item()
        batches += 1
        score_list.append(torch.sigmoid(outputs["anomaly_logits"]))
        label_list.append(batch["anomaly_target"])

    scores = torch.cat(score_list)
    labels = torch.cat(label_list)
    pr_auc = binary_pr_auc(scores, labels)
    best_f1, threshold, precision, recall = best_f1_threshold(scores, labels)
    return {
        "val_loss": total_loss / max(batches, 1),
        "forecast_mse": total_forecast_loss / max(batches, 1),
        "anomaly_bce": total_anomaly_loss / max(batches, 1),
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_threshold": threshold,
        "precision": precision,
        "recall": recall,
        "positive_rate": labels.float().mean().item(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local time-series dataset for autoresearch-ts-anomaly")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to a .npz or .csv dataset")
    parser.add_argument("--num-steps", type=int, default=8000, help="Number of time steps for synthetic data generation")
    parser.add_argument("--num-features", type=int, default=DEFAULT_FEATURE_DIM, help="Number of features for synthetic data generation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        generate_synthetic_dataset(dataset_path, num_steps=args.num_steps, num_features=args.num_features)

    values, labels = load_raw_dataset(dataset_path)
    positives = int(labels.sum())
    print(f"dataset_path:      {dataset_path}")
    print(f"time_steps:        {len(values)}")
    print(f"num_features:      {values.shape[1]}")
    print(f"positive_labels:   {positives}")
    print(f"positive_ratio:    {positives / max(len(labels), 1):.4f}")
    print(f"context_len:       {CONTEXT_LEN}")
    print(f"prediction_horizon:{PREDICTION_HORIZON}")


if __name__ == "__main__":
    main()
