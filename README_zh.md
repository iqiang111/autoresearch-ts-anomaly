# autoresearch-ts-anomaly

[English README](./README.md)

这是一个从 `karpathy/autoresearch` 分叉出来的最小化仓库，目标是服务于时间序列异常预测研究。

这个分叉保留了原项目“小仓库、固定时间预算、便于代理自动实验”的工作流，但把原来的语言建模任务替换成了滑动窗口时间序列任务：

- 输入：过去 `context_len` 个时间步的多变量数值
- 输出：
  - 未来 `prediction_horizon` 个时间步的预测值
  - 未来窗口内是否出现异常的概率
- 验证指标：
  - `val_loss`
  - `forecast_mse`
  - `anomaly_bce`
  - `pr_auc`
  - `best_f1`

## 主要改动

原始仓库默认绑定的是文本分词、GPT 预训练和 `val_bpb`。这个分叉替换成了：

- `prepare.py`：本地时间序列数据加载、训练/验证/测试划分、归一化、滑窗构造、合成数据兜底、异常指标评估
- `train.py`：一个紧凑的 Transformer 基线，同时做未来值预测和异常分类
- `program.md`：面向异常预测实验的代理说明，而不是 LLM 预训练说明

## 数据格式

默认数据路径是 `data/synthetic_timeseries.npz`。如果文件不存在，运行 `uv run prepare.py` 会自动生成一个带注入异常的合成多变量时间序列数据集。

当前支持两种格式：

- `.npz`
  - `values`：形状为 `[time, features]`
  - 可选 `labels`：形状为 `[time]`，二值异常标签
- `.csv`
  - 数值特征列
  - 可选标签列，列名可为：
    - `label`
    - `labels`
    - `anomaly`
    - `target`

如果没有标签，`prepare.py` 会根据一阶差分的突变自动构造弱监督伪标签。这个机制只适合做启动测试，不适合作为正式研究的标注来源。

## 快速开始

环境要求：Python 3.10+、PyTorch、`uv`。

```bash
uv sync
uv run prepare.py
uv run train.py
```

训练脚本默认运行 `TIME_BUDGET=300` 秒，并在结束时输出摘要指标。

如果你使用当前机器上的 `miniconda`，建议统一通过 `price` 环境执行：

```bash
C:\Users\22077\miniconda3\Scripts\conda.exe run -n price python prepare.py
C:\Users\22077\miniconda3\Scripts\conda.exe run -n price python train.py
```

## 项目结构

```text
prepare.py      数据准备、划分逻辑、dataloader、评估
train.py        基线 Transformer 与训练循环
program.md      自动研究说明
pyproject.toml  依赖定义
```

## 研究用途

这个仓库是自动化实验的起点，不是一个已经完整封装好的异常检测 benchmark 框架。下一步通常包括：

1. 用你的真实领域数据替换当前合成数据。
2. 把基线模型替换成更适合你任务的 patch、graph、diffusion 或 state-space 结构。
3. 如果你的异常是区间事件而不是点异常，增加 event-level 指标。
4. 扩展 `program.md`，让代理围绕 `pr_auc` 或你的自定义目标做实验选择。

## 许可证

MIT
