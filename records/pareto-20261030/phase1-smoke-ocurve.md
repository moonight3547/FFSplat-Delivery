# Phase-1 smoke test: existing data only (NOT the E2 result)

## 0. 输入与覆盖

- 结果文件：`['records/pareto-20261030/inputs/ocurve']`
- 13 例串行锚表：`records/pareto-20261030/inputs/scene13r15-results.jsonl`
- 网格：[500, 1000, 2000, 3000, 5000, 8000]
- 场景数：3；行数：7；有效行：7
- **策略对比集合**：2/3 个场景（anchor-only；只有该集合内的均值可直接比较）
- **缺失格子（显式列出，不猜测）：** dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce@1000, dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce@3000, dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce@8000, dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e@1000, dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e@3000, dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e@8000, dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3@1000, dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3@2000, dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3@3000, dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3@5000, dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3@8000

## 1. 每场景 Pareto 表

标记：`◀` = 该场景 total argmax；`·` = quality-vs-B 非支配点；`INVALID` = B ≥ budget（按 AGENTS.md 从均值中剔除）。

| case | S | quality | B(s) | TIME | total | 标记 |
|---|---:|---:|---:|---:|---:|---|
| dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce | 500 | 5.19 | 12.48 | 19.28 | 24.47 | · |
| dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce | 2000 | 10.67 | 18.49 | 17.86 | 28.54 | · |
| dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce | 5000 | 17.06 | 36.05 | 13.73 | 30.79 | ◀ · |
| dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e | 500 | 3.78 | 10.33 | 19.79 | 23.57 | · |
| dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e | 2000 | 6.02 | 17.58 | 18.08 | 24.10 | · |
| dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e | 5000 | 17.82 | 36.57 | 13.60 | 31.42 | ◀ · |
| dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3 | 500 | 5.41 | 9.85 | 19.90 | 25.31 | ◀ · |

### 1b. 每场景摘要

| case | argmax S | quality@argmax | B@argmax | total@argmax | total@S=5000 | Δ vs 5000 | frontier S |
|---|---:|---:|---:|---:|---:|---:|---|
| dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce | 5000 | 17.06 | 36.05 | 30.79 | 30.79 | 0.00 | 500, 2000, 5000 |
| dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e | 5000 | 17.82 | 36.57 | 31.42 | 31.42 | 0.00 | 500, 2000, 5000 |
| dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3 | 500 | 5.41 | 9.85 | 25.31 | - | - | 500 |

## 2. 策略对比（集合内均值，fixed anchor S=5000）

| 策略 | S | n/集合 | 集合内均值 total | 相对 (a) |
|---|---|---:|---:|---:|
| (a) fixed S=5000 (current delivery) | - | 2/2 | 31.107 | - |
| (b) best single uniform S | 5000 | 2/2 | 31.107 | 0.000 |
| (c) per-scene oracle (DIAGNOSTIC ceiling only) | per-scene | 2/2 | 31.107 | 0.000 |
| (d) marginal-value stop simulated on TEST curves (optimistic, NOT deployable) | per-scene | 2/2 | 31.107 | 0.000 |

### 2b. 每个 S 的均匀策略均值

| S | n/场景数 | 均值 total | 均值 quality | 均值 B(s) | INVALID 场景 |
|---:|---:|---:|---:|---:|---|
| 500 | 2/2 | 24.019 | 4.48 | 11.40 | - |
| 1000 | 0/2 | - | - | - | - |
| 2000 | 2/2 | 26.319 | 8.35 | 18.04 | - |
| 3000 | 0/2 | - | - | - | - |
| 5000 | 2/2 | 31.107 | 17.44 | 36.31 | - |
| 8000 | 0/2 | - | - | - | - |

## 3. 边际停止规则（grid marginal-value stop）

阈值 λ = 20/(0.9·budget)，取值区间 [0.2357]

> 这里的 (d) 用**测试集质量曲线**模拟停止规则，是「完美信息」的乐观上界；可部署版本必须改用非 context 的 train 视图，见 PARETO-STEPS 文档 §5。

| case | budget | λ | 规则选 S | 规则 total | oracle S | oracle total | 区间决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce | 94.29 | 0.236 | 5000 | 30.790 | 5000 | 30.790 | 500→2000: Δq/ΔB=0.91; 2000→5000: Δq/ΔB=0.36 |
| dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e | 94.29 | 0.236 | 5000 | 31.424 | 5000 | 31.424 | 500→2000: Δq/ΔB=0.31; 2000→5000: Δq/ΔB=0.62 |
| dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3 | 94.29 | - | - | - | - | - | 无有效点 |

## 5. 13 例串行锚（H1 `scene13r15/results.jsonl`）

- baseline O0 均值 total = **41.371**（n=13）
- delivery S=5000 均值 total = **49.599**（n=13）
- Δ = 8.228

- 扫描表 S=5000 均值 total = 31.107；与串行锚差 -18.492 （若明显偏低，先怀疑并发把 B 抬高，而不是配方变化）

## 6. 方差与稳健性

- best-uniform vs fixed-5000 差 +0.000 分 < 1 分：属噪声量级，需要 ≥3 次重复才能裁决（当前 n=2 场景单次）。
- oracle 比 best-uniform 高 +0.000 分：这是「允许逐场景选 S」的诊断上界，按 AGENTS.md 不可作为提交策略。
- 2/2 个场景的 oracle 增益 vs S=5000 小于 1 分（[0.0, 0.0]）：这些场景的 S 决策落在噪声内，至少 3 次重复才可裁决。
- TIME 敏感度：预算 94.29s 时每 1s 模块 B ≈ 0.236 分；中位 B=17.6s 的 ±45% 并发膨胀 ≈ ±1.86 分，足以翻转 <2 分的策略差异——并发 B 只能用于质量，不能用于策略排序。
- 每场景 argmax S 分布：[500, 5000]（中位 5000）。
- 扫描 S=5000 均值 31.107 vs 13 例串行锚 49.599（Δ -18.492）：可作为并发偏差方向的交叉校验。

## 7. 警告

- not every scene has a complete grid; policy comparison uses only the 2 scenes with a valid S=5000 anchor (out of 3). This is a degraded comparison.
