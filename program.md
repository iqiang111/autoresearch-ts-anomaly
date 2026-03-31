# autoresearch-ts-anomaly

This repo is an experiment in autonomous research for time-series anomaly prediction.

## Setup

To set up a new experiment:

1. Read `README.md`, `prepare.py`, and `train.py`.
2. Check whether a dataset exists under `data/`.
3. If not, run `uv run prepare.py` to generate the synthetic fallback dataset.
4. Initialize `results.tsv` with:

```text
commit	pr_auc	best_f1	memory_gb	status	description
```

5. Run a baseline before changing anything.

## Task definition

Each training example is:

- input: past `context_len` time steps
- targets:
  - future `prediction_horizon` values
  - binary label indicating whether the future window contains an anomaly

The script trains for a fixed wall-clock budget of 5 minutes by default.

## Research objective

Primary metric: maximize `pr_auc`.

Secondary metrics:

- maximize `best_f1`
- minimize `val_loss`
- avoid unnecessary complexity
- keep memory use reasonable

If two runs are close, prefer the simpler change.

## What you can edit

- `train.py`: model architecture, losses, optimizer, schedules, batching, training loop
- `program.md`: human-authored research policy
- `prepare.py`: only when dataset format, split logic, or evaluation truly needs to change

## What you should not do casually

- Do not change evaluation just to make metrics look better.
- Do not add dependencies unless the human explicitly asks.
- Do not keep large generated datasets under version control.

## Logging results

Record each experiment in `results.tsv`:

```text
commit	pr_auc	best_f1	memory_gb	status	description
```

Example:

```text
commit	pr_auc	best_f1	memory_gb	status	description
a1b2c3d	0.412300	0.381000	3.2	keep	baseline transformer
b2c3d4e	0.447200	0.401500	3.4	keep	increase model dim to 192
c3d4e5f	0.398100	0.352900	3.1	discard	remove anomaly head hidden layer
d4e5f6g	0.000000	0.000000	0.0	crash	invalid tensor reshape
```

## Experiment loop

1. Inspect current git state.
2. Make one concrete experimental change.
3. Commit it.
4. Run `uv run train.py > run.log 2>&1`.
5. Read the summary metrics from `run.log`.
6. If the run crashes, inspect the tail of the log and fix obvious issues.
7. Append the result to `results.tsv`.
8. Keep the commit only if `pr_auc` improves materially, or if a near-tie is clearly simpler and safer.

## Output summary

The current script prints:

```text
---
val_loss:
forecast_mse:
anomaly_bce:
pr_auc:
best_f1:
best_threshold:
precision:
recall:
positive_rate:
training_seconds:
peak_vram_mb:
num_steps:
num_params_M:
```
