# autoresearch-ts-anomaly

Minimal fork of `karpathy/autoresearch` for time-series anomaly prediction research.

This fork keeps the small-repo, fixed-time-budget, agent-friendly workflow, but swaps the original language-modeling task for a sliding-window time-series task:

- input: past `context_len` multivariate values
- outputs:
  - future `prediction_horizon` values
  - probability that the future window contains an anomaly
- validation metrics:
  - `val_loss`
  - `forecast_mse`
  - `anomaly_bce`
  - `pr_auc`
  - `best_f1`

## What changed

The original repository was hard-wired to text tokenization, GPT pretraining, and `val_bpb`. This fork replaces that with:

- `prepare.py`: local time-series loading, train/val/test split, normalization, windowing, synthetic fallback dataset, anomaly metrics
- `train.py`: compact Transformer baseline for forecasting plus anomaly classification
- `program.md`: agent instructions tuned for anomaly-prediction experiments instead of LLM pretraining

## Dataset format

The default path is `data/synthetic_timeseries.npz`. If the file does not exist, `uv run prepare.py` creates a synthetic multivariate dataset with injected anomalies.

Supported formats:

- `.npz` with:
  - `values`: shape `[time, features]`
  - optional `labels`: shape `[time]`, binary anomaly labels
- `.csv` with numeric feature columns and an optional label column named one of:
  - `label`
  - `labels`
  - `anomaly`
  - `target`

If labels are missing, `prepare.py` derives weak pseudo-labels from large first-difference spikes. That fallback is only for bootstrapping; real experiments should provide labels.

## Quick start

Requirements: Python 3.10+, PyTorch, `uv`.

```bash
uv sync
uv run prepare.py
uv run train.py
```

The training script runs for `TIME_BUDGET=300` seconds by default and prints a compact summary at the end.

## Project structure

```text
prepare.py      data preparation, split logic, dataloaders, evaluation
train.py        baseline Transformer and training loop
program.md      autonomous research instructions
pyproject.toml  dependencies
```

## Research use

This repo is a starting point for automated experimentation, not a finished anomaly benchmark framework. Useful next steps include:

1. Replacing the synthetic dataset with your own domain data.
2. Swapping the baseline model for patching, graph, diffusion, or state-space variants.
3. Using event-level metrics if your anomalies are interval-based rather than point-based.
4. Extending `program.md` so agents compare changes against `pr_auc` or your own objective.

## License

MIT
