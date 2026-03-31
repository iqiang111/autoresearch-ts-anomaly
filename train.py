"""
Autoresearch baseline for time-series anomaly prediction.

Usage:
    uv run train.py
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    CONTEXT_LEN,
    DEFAULT_BATCH_SIZE,
    PREDICTION_HORIZON,
    TIME_BUDGET,
    evaluate_model,
    make_dataloader,
    prepare_data,
)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

MODEL_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_RATIO = 4
DROPOUT = 0.1
BATCH_SIZE = DEFAULT_BATCH_SIZE
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
ANOMALY_LOSS_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 25
SEED = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    context_len: int
    prediction_horizon: int
    num_features: int
    model_dim: int = MODEL_DIM
    num_heads: int = NUM_HEADS
    num_layers: int = NUM_LAYERS
    mlp_ratio: int = MLP_RATIO
    dropout: float = DROPOUT


class PositionalEncoding(nn.Module):
    def __init__(self, context_len: int, model_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, context_len, model_dim))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_ratio * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * model_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TimeSeriesAnomalyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.num_features, config.model_dim)
        self.positional = PositionalEncoding(config.context_len, config.model_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.model_dim, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.model_dim)
        self.forecast_head = nn.Linear(config.model_dim, config.prediction_horizon * config.num_features)
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_proj(context)
        x = self.positional(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = x[:, -1]
        future = self.forecast_head(pooled).view(
            context.size(0), self.config.prediction_horizon, self.config.num_features
        )
        anomaly_logits = self.anomaly_head(pooled).squeeze(-1)
        return {
            "future_values": future,
            "anomaly_logits": anomaly_logits,
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_losses(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(batch["context"])
    forecast_loss = F.mse_loss(outputs["future_values"], batch["future_values"])
    anomaly_loss = F.binary_cross_entropy_with_logits(outputs["anomaly_logits"], batch["anomaly_target"])
    total_loss = forecast_loss + ANOMALY_LOSS_WEIGHT * anomaly_loss
    return total_loss, forecast_loss, anomaly_loss


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    device = get_device()
    prepared = prepare_data(context_len=CONTEXT_LEN, prediction_horizon=PREDICTION_HORIZON)
    train_loader = make_dataloader(
        prepared.train,
        batch_size=BATCH_SIZE,
        context_len=CONTEXT_LEN,
        prediction_horizon=PREDICTION_HORIZON,
        shuffle=True,
        device=device,
    )
    val_loader = make_dataloader(
        prepared.val,
        batch_size=BATCH_SIZE,
        context_len=CONTEXT_LEN,
        prediction_horizon=PREDICTION_HORIZON,
        shuffle=False,
        device=device,
    )

    config = ModelConfig(
        context_len=CONTEXT_LEN,
        prediction_horizon=PREDICTION_HORIZON,
        num_features=prepared.num_features,
    )
    model = TimeSeriesAnomalyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"device:            {device}")
    print(f"model_config:      {asdict(config)}")
    print(f"time_budget:       {TIME_BUDGET}")
    print(f"batch_size:        {BATCH_SIZE}")
    print(f"train_windows:     {len(train_loader.dataset)}")
    print(f"val_windows:       {len(val_loader.dataset)}")

    model.train()
    start_time = time.time()
    steps = 0
    smooth_loss = 0.0
    warm_steps = 3

    while True:
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            total_loss, forecast_loss, anomaly_loss = compute_losses(model, batch)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            loss_value = total_loss.detach().item()
            smooth_loss = 0.95 * smooth_loss + 0.05 * loss_value
            debiased = smooth_loss / (1.0 - 0.95 ** (steps + 1))
            steps += 1

            elapsed = time.time() - start_time
            if steps % LOG_EVERY == 0:
                print(
                    f"step {steps:05d} | "
                    f"loss {debiased:.4f} | "
                    f"forecast_mse {forecast_loss.item():.4f} | "
                    f"anomaly_bce {anomaly_loss.item():.4f} | "
                    f"elapsed {elapsed:.1f}s"
                )

            if steps > warm_steps and elapsed >= TIME_BUDGET:
                break
        if steps > warm_steps and (time.time() - start_time) >= TIME_BUDGET:
            break

    training_seconds = time.time() - start_time
    metrics = evaluate_model(model, val_loader, anomaly_loss_weight=ANOMALY_LOSS_WEIGHT)
    num_params = sum(p.numel() for p in model.parameters())
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    )

    print("---")
    print(f"val_loss:          {metrics['val_loss']:.6f}")
    print(f"forecast_mse:      {metrics['forecast_mse']:.6f}")
    print(f"anomaly_bce:       {metrics['anomaly_bce']:.6f}")
    print(f"pr_auc:            {metrics['pr_auc']:.6f}")
    print(f"best_f1:           {metrics['best_f1']:.6f}")
    print(f"best_threshold:    {metrics['best_threshold']:.6f}")
    print(f"precision:         {metrics['precision']:.6f}")
    print(f"recall:            {metrics['recall']:.6f}")
    print(f"positive_rate:     {metrics['positive_rate']:.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"num_steps:         {steps}")
    print(f"num_params_M:      {num_params / 1e6:.2f}")


if __name__ == "__main__":
    main()
