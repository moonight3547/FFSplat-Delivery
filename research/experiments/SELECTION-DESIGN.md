# COVER / PUN 路线的实验设计（先诊断，后花 GPU）

## 关键前置发现（必须先说）

`research/advisory/VIEW_SELECTION_FEASIBILITY_20261030.md` 早就指出「prepared work 没有真实 tracks」。
刚在 N1 上核实：`work/sparse/0/` 里**只有 `cameras.txt` + `images.txt`，没有 `points3D.txt`**。
也就是说，**基于 track 匹配支持的选图指标在现有产物上根本算不出来**。

因此 COVER 类方法的验证必须先补一步：**对原始数据跑一次 COLMAP 三角化**（hw 类已有 sparse 模型；
DL3DV/mip 只有 transforms.json，需要先 COLMAP 重建再三角化）。这一步是额外的 A 侧成本，
不能假装它免费——而且它只用于**诊断**，不进交付流程。

## 诊断阶段（CPU，先用 0 张 GPU）

对每个候选选集 S（uniform / COVER 式覆盖 / fps），计算：

| 指标 | 定义 | 用途 |
|---|---|---|
| `support2` | 在 S 中被 ≥2 个视角观测到的 3D track 数 | 匹配支持的核心量 |
| `support1` | 被至少 1 个视角观测到的 track 数 | 覆盖广度 |
| `obs_per_track` | 平均每 track 的观测数 | 匹配冗余 |
| `novel_ratio` | 仅被 1 个视角观测的 track 占比 | 弱匹配风险 |

**判定规则（先确认候选真的改变了匹配支持，再上 GPU）**：
- bicycle（已知难例）：候选的 `support2` 必须比 uniform **高 ≥10%**，否则不看质量；
- counter（已知好例）：`support2` **不得低于 uniform 的 95%**（不允许用质量换覆盖）；
- 一个**无可验证 track** 的场景：必须能正确回退到 uniform，且记录回退原因。

## 质量阶段（只有在诊断通过后）

3 个场景 × 2 个选集（uniform / 候选）→ ReSplat + 交付配方 → 官方指标。
门槛沿用之前：难例总分 **+2** 以上、好例不得回退超过 1 分。

## 当前不采用的做法

- 不搬 COVER 原流程（它依赖已有重建、逐步加视图，与我们"一次前馈固定 16 帧"的预算模型不符）；
- 不搬 PUN/UPNet（物体级主动观察，未见长轨迹室内/室外证据）；
- 不在没有 track 数据的情况下用"相机包围盒覆盖"冒充 COVER——那正是我们之前失败过的 coverage 策略。
