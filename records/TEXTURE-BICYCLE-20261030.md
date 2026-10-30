# Bicycle-only 增密实验（距离 / 空洞 / 误差密度）：全负结论（2026-10-31，N1）

> **运行环境：`# GPU2 / shared box`**。sweep 时段 N1 是共享机（teammate staged-loss 在
> GPU0 跑自己的 step-sweep），所以本文件的 B/TIME 按"受争用"记录；`control` 与最终裁决
> 用的 `g10d` 另在 **GPU0 于 02:25–02:38 的安静窗口**重测（`__quiet` 后缀），两套 B 的
> 差值 ≤0.1 s（control 27.2 vs 27.1），因此结论不依赖争用假设。
> 前置报告：`TEXTURE-VOXEL-20261030.md`（task-7：裁剪+增密；四变体全部 drop）。
> **没有任何变体被采纳；交付配方保持 `--init-ply = 原始 resplat_world.ply`。**

---

## 1. 【结论先行】α 修正版训练视角诊断：init 是"全覆盖但糊"，不是"外围/空洞欠 GS"

`research/init_error_diag.py`：与模块 B 相同的 FastGS 光栅器，6 个固定**训练视角**各渲染
两次（白底/黑底），得精确累计 α：`α = 1 − mean(img_white − img_black)`；与训练 GT 求平方误差；
用"最近高斯中心 z-buffer + EDT 最近邻填充"得逐像素深度代理，反投影后按
`‖x−scene_center‖/scene_radius` 分桶（0.5–99.5% 分位包围盒，与 `ply_voxgrow.py` 同定义）。
运行：`resume.sh diag 2`，GPU2，02:15:14，9 秒，IDLE-GATE PASS。

| 指标（α 修正版，**以本节为准**） | 数值 |
|---|---|
| 采样视角（全部为 `split.json.train`，脚本硬断言，含 test 视角即 `SystemExit`） | `_DSC8815, _DSC8690, _DSC8855, _DSC8835, _DSC8869, _DSC8793` |
| 平均 α | **0.9964**（逐视角 0.9901–0.9998） |
| 像素占比 α<0.5 | **0.0000** |
| 像素占比 α<0.1 | **0.0000** |
| α<0.5 像素贡献的平方误差占比 | **0.0000** |
| 天空/场景外像素（最近 α≥0.5 像素 >48 px） | 0.0% |
| 3D 空洞点数（hole cloud） | **0**（无可发射点，文件未生成） |
| 训练视角渲染 PSNR（原始 init） | 15.77 |
| 外围（>0.5R）高斯占比 | 8.74% |
| 外围（>0.5R）平方误差占比 | **6.52%** |
| 外围（>0.5R）像素占比 | 11.5% |
| **误差/高斯比（>0.5R）** | **0.75** |
| 训练视角导出误差点（M,4） | 3 317 760 |

**结论：原始 ReSplat init 没有任何未覆盖像素**（52 万高斯、每个投影约 10–15 px，
把所有训练视角铺满），因此"空洞"假设与"外围欠 GS"假设同时被否定；
v1 的 `uncovered_pixel_frac=0.802 / uncovered_error_share=0.442` **是中心 z-buffer 伪影，正式撤回**（见 §1.4）。

### 1.1 按半径分桶（α≥0.5 像素，深度用 EDT 填充代理）——距离先验方向相反的最强证据

| 半径桶 | 高斯占比 | 平方误差占比 | 像素占比 | 逐像素 MSE |
|---|---:|---:|---:|---:|
| 0–0.25 R | 50.49% | 48.60% | 46.30% | **0.0278** |
| 0.25–0.5 R | 40.77% | 44.88% | 42.20% | **0.0282** |
| 0.5–0.75 R | 6.82% | 5.51% | 9.92% | 0.0147 |
| 0.75–1 R | 1.24% | 0.75% | 1.29% | 0.0154 |
| >1 R | 0.69% | 0.26% | 0.29% | 0.0235 |

逐像素误差在**内圈最高**（0.0278/0.0282），在中环最低（0.0147/0.0154）。
任何"越远权重越大"的距离先验都是在**瞄准误差更小的区域**——这是本轮所有距离变体
质量持平的直接原因，而不是实现细节。

### 1.2 误差的空间集中度（同一 64³ 网格）

| 体素集合 | 占误差质量 |
|---|---:|
| 误差最高的 1% 体素 | **37.3%** |
| 最高 5% | 68.8% |
| 最高 10% | **82.2%** |
| 最高 25% | 94.4% |
| 最高 50% | 98.9% |

误差**有**集中性（10% 体素承载 82% 误差），所以"按误差密度增密"是本轮唯一有真实目标的机制
（§2 g10d）。注意：误差点云由 EDT 填充深度反投影得到，其中只有 8 073/20 827 个误差体素与
高斯体素重合（≈39%），其余落在无高斯的空间——这是深度代理的偏差，报告为**已知局限**，
不影响"误差不在外围/没有空洞"的结论（那两条只用 α 与已覆盖像素）。

### 1.3 审计链（train-view-only，无测试泄漏）

1. 相机来自 `scene.getTrainCameras()`；脚本无 `getTestCameras()` 代码路径。
2. 选点后硬断言：6 个视角的 `image_name` 必须都在 `<work>/split.json.train`，任一在
   `test` 中即 `SystemExit`。断言通过（见上表视角名）。
3. GT 用训练视角 `cam.original_image`；不读测试渲染/测试 GT。
4. α 由两次同视角渲染（白/黑底）闭合得到，与背景色无关。
5. 输出 JSON 含 `train_views_only: true`、`camera_names`、`hole_meta`（α 阈值、48 px 天空排除规则）。
6. 机制对任意场景统一（渲染 K 个训练视角 → α → 3D 点 → 体素权重），不做逐场景调参、不用测试集选点。

### 1.4 v1 数字（**已撤回，仅供审计，不要引用**）

| 指标 | v1（伪影版） | α 修正版 |
|---|---:|---:|
| uncovered 像素占比 | 0.802 | **0.0000**（α<0.5） |
| uncovered 对误差贡献 | 0.442 | **0.0000** |
| 渲染 PSNR（4 视角） | 16.37 | 15.77（6 视角） |
| 高斯占比 >0.5R | 8.74% | 8.74% |
| 误差占比 >0.5R | 7.30% | **6.52%** |
| 误差/高斯比 >0.5R | 0.83 | **0.75** |

v1 的 44%/80% 来自"没有高斯**中心**落在该像素 ≠ 空洞"（bicycle 高斯投影 ~10–15 px），
已全部作废；唯一保留的是"外围误差占比低于外围高斯占比"这一方向（α 修正版更强）。

## 2. 本轮测试的增密机制（全部 grow-only，无裁剪）

统一：`--op grow_distonly --no-prune --grow-frac 0.2 --voxels 64`，只换**体素权重**；
每个子高斯 = 随机父高斯复制（scale ÷1.6，位置抖动 0.3σ）；变换 0.6–1.1 s/PLY。

| arm | 权重 | 说明 |
|---|---|---|
| `g1_hard05` | 1×，d>0.5R 时 2×；闸门 count≤1 | 用户原始"外围 2 倍"想法（无裁剪 + 可判别闸门） |
| `g2_ramp2` | 1 + 1·(d/R)² 连续 ramp；闸门 count≤1 | 连续距离先验 |
| `g8_inner` | d≤0.25R 内均匀，外侧 0；无闸门 | 诊断给出的"反方向"符号检验（误差在内圈） |
| `g10d_errgrow` | 每体素 = 训练视角平方误差累计（误差点云）；cap=128/体素 | 唯一有真实目标的机制（§1.2） |
| `g9_holes` | 每体素 = 空洞点数 | **取消**：α<0.5 像素为 0，空洞点云为空，无目标 |
| ~~g10/g10b/g10c~~ | ~~errdensity，cap=1/8/128~~ | **实现 bug 作废**：距离先验覆盖了误差权重（当时三者的权重实际是"误差体素→2×"硬先验，PLY 与 g1 同类）；修复后才有 g10d |

**"欠高斯"闸门**（此轮仍用 count≤1，因为诊断显示 init 全覆盖，用不到更激进的闸门）：
在 64³ 网格上 8 957/28 258 个有高斯体素满足 count≤1（31.7%），每个体素允许 16 个子高斯
（预算装不下时自动放宽上限）。

### 2.1 新增子高斯落点（可解释性）

| arm | 子高斯 >0.25R | >0.5R | >0.75R | 平均 d/R（全部→子高斯） | 误差最高 10% 体素中的子高斯占比（高斯基线 34.9%） |
|---|---:|---:|---:|---:|---:|
| `g1_hard05` | 90.5% | 61.6% | 10.5% | 0.295 → **0.533** | — |
| `g2_ramp2` | 88.6% | 50.8% | 10.1% | 0.295 → **0.507** | — |
| `g8_inner` | 0% | 0% | 0% | 0.295 → **0.171** | — |
| `g10d_errgrow` | 71.6% | 10.7% | 1.9% | 0.295 → 0.338 | **66.1%** |

权重**确实**把新基元放到了设计位置（g1/g2 把 61.6%/50.8% 的新基元放到 >0.5R，
而全场景只有 8.8% 的高斯在那里；g10d 把 66.1% 放到误差最高的 10% 体素，基线 34.9%），
所以下面的质量/D 结论不是"权重没生效"造成的。

## 3. 结果（bicycle，模块 B 配方不变，S=5000）

`q` = PSNR/SSIM/LPIPS 代理质量分；`B` = 模块 B 秒（含 scene load/init/train，预算 95.0）；
`TIME` = 20·min(1,(1−B/95)/0.9)；`total` = q + TIME。n_out = 625 176（+20%），
n_train = 训练结束后的高斯数（模块 B 会继续致密化）。

| arm（运行环境） | n_train | PSNR | SSIM | LPIPS | q | B(s) | TIME | total | Δq* | ΔB* | Δtotal* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control（**quiet** GPU0） | 1 003 368 | 24.55 | 0.730 | 0.297 | 10.05 | 27.1 | 15.89 | **25.94** | — | — | — |
| control（contended GPU2） | 1 005 632 | 24.58 | 0.730 | 0.298 | 10.04 | 27.2 | 15.86 | 25.90 | −0.01 | +0.1 | −0.04 |
| g1_hard05 | 1 120 338 | 24.58 | 0.729 | 0.298 | 10.01 | 28.9 | 15.45 | 25.46 | **−0.04** | +1.9 | **−0.48** |
| g2_ramp2 | 1 119 285 | 24.55 | 0.730 | 0.298 | 10.07 | 30.0 | 15.21 | 25.28 | +0.02 | +2.9 | **−0.66** |
| g8_inner | 1 131 527 | 24.55 | 0.731 | 0.295 | 10.11 | 28.8 | 15.49 | 25.60 | +0.07 | +1.7 | **−0.34** |
| g10d_errgrow（误差密度） | 1 111 155 | 24.58 | 0.732 | **0.294** | **10.18** | 30.2 | 15.16 | 25.34 | **+0.13** | +3.1 | **−0.60** |

\* Δ 以 quiet control 为基线；质量噪声 ±0.2（task-7 同配置 parallel vs serial 差 0.03/0.19）。

**质量是平的**：−0.04 ~ +0.13，全部落在 ±0.2 噪声带内（误差密度臂 +0.13 是最大值且方向与
诊断一致，但仍不可区分于噪声）；**B 全部 +1.7 ~ +3.1 s**，TIME 损失 0.41–0.73 分，
**每一个变体 total 都低于 control**，没有一个满足 "serial total > control + 0.5"。

`g10d` 的 +3.1 s 可能含邻居作业的 CPU 争用（运行窗口 GPU2 上有 leased 作业）；即便按
同族 g10 的 +2.3 s 计，total 也只能到 ≈25.47，仍 −0.47 —— 结论不变。

## 4. B / TIME 权衡曲线（bicycle，quiet control = 25.94）

| arm | Δq | ΔB (s) | ΔTIME | Δtotal |
|---|---:|---:|---:|---:|
| g1_hard05 | −0.04 | +1.9 | −0.44 | −0.48 |
| g2_ramp2 | +0.02 | +2.9 | −0.68 | −0.66 |
| g8_inner | +0.07 | +1.7 | −0.41 | −0.34 |
| g10d_errgrow | +0.13 | +2.3~3.1 | −0.55~−0.73 | −0.47~−0.60 |

每 20% 初始基元 ≈ **+1.7~3.1 s B**（模块 B 终态从 1.003 M 涨到 1.11–1.13 M），
对应 TIME **−0.41~−0.73**。要打平，20% 增密必须带来 **≥0.4~0.7 质量分**；
本轮最好的机制只有 +0.13。**结论：在这个预算/模块 B 下，init 端 +20% 增密的时间成本
无法被这些空间权重赚回来。**

## 5. 裁决

| 变体 | 判定 | 依据 |
|---|---|---|
| control（原始 PLY） | **保留** | — |
| `g1_hard05`（外围 2×） | **丢弃** | Δq −0.04，Δtotal −0.48；且诊断显示外围误差占比 < 外围高斯占比 |
| `g2_ramp2`（(d/R)²） | **丢弃** | Δq +0.02，Δtotal −0.66 |
| `g8_inner`（内圈生长） | **丢弃** | Δq +0.07（方向对但太小），Δtotal −0.34 |
| `g10d_errgrow`（误差密度） | **丢弃** | 唯一质量正号（+0.13）但 B +2.3~3.1 s，Δtotal −0.47~−0.60 |
| `g9_holes`（空洞） | **不成立** | α<0.5 像素为 0，无空洞可补 |

**最终结论（三机制齐闭）**：`init 端空间再分配` 这条路对总分无益——
(1) 误差不在外围（α 修正后 >0.5R 误差 6.52% < 高斯 8.74%，且内圈逐像素误差最高）；
(2) 没有未覆盖像素（α<0.5 = 0.0000），"欠 GS/空洞"不成立；
(3) 四种空间权重（距离 2×、连续 ramp、内圈、误差密度）质量全部 ±0.1 级，
而每个 +20% 都要付 1.7–3.1 s 的 B。模块 B 自己 5000 步会重新致密化并修正颜色/几何，
init 端的密度重分配既信息不足又很贵。
**建议**：交付保持 `--init-ply = 原始 resplat_world.ply`，不再在 init 端做空间重分配。

唯一的"有信号"方向（供将来参考，未验证）：**误差密度**（g10d）是本轮唯一把质量推正的
权重（+0.13），但需要把 20% 增密的时间成本压到 +1 s 以内才可能打平——例如只对产生
最大误差的 1% 体素做少量分裂（更稀疏、更便宜），而不是全局 +20%。

## 6. 精确命令

```bash
# 0) 空闲闸门 + α 诊断（GPU2）= resume.sh diag 2；再跑串行 = resume.sh serial 2
#    resume.sh: idle 检查（目标卡 <1500MiB / 无 compute app / 无我们自己的诊断进程）
bash /root/gpufree-data/outputs/bicycle-grow-20261030/resume.sh diag 2

# 1) α 诊断（训练视角，6 views × 2 renders）→ α 统计 + 误差点云（空洞点云为空）
cd /root/projects/FFSplat-Hackathon && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 \
  /opt/conda/envs/fastgs/bin/python -m research.init_error_diag \
  --source outputs/prep/mipnerf360/bicycle/work \
  --ply outputs/prep/mipnerf360/bicycle/resplat/resplat_world.ply --views 6 \
  --out /root/gpufree-data/outputs/bicycle-grow-20261030/init_error_diag_alpha.json \
  --out-holes /root/gpufree-data/outputs/bicycle-grow-20261030/holes_bicycle.npy \
  --out-error-points /root/gpufree-data/outputs/bicycle-grow-20261030/error_points_bicycle.npy

# 2) 变换（grow-only，64³，权重按 arm 选择；~0.6–1.1 s/PLY）
python -m research.ply_voxgrow --in <init.ply> --out g10d.ply \
  --op grow_distonly --no-prune --grow-frac 0.2 --voxels 64 --under-mode none \
  --dist-mode errdensity --error-points error_points_bicycle.npy --cap-scale 128 \
  --summary-out g10d.json
#   其它 arm：g1 = --under-mode le1 --dist-mode hard --far-frac 0.5 --far-weight 2
#             g2 = --under-mode le1 --dist-mode ramp2
#             g8 = --under-mode none --dist-mode inneronly --far-frac 0.25

# 3) 模块 B：交付配方不变，只换 init PLY；串行单卡（profile gpufree-<GPU_LANE>）
python -m research.run_sweep --profile competition/profiles/gpufree-0.json \
  --case mipnerf360/bicycle --work outputs/prep/mipnerf360/bicycle/work \
  --init-ply g10d.ply --out <out> --iterations 5000 --snapshots "" --budget 95.0 \
  --position-lr-max-steps 5000 --opacity-reset-interval 0 --lambda-dssim 0.6 \
  --scaling-lr 0.01 --densify-from 500 --densify-until 1250 \
  --realloc-frac 0.15 --realloc-at 1250 \
  --image-cache outputs/prep/mipnerf360/bicycle/work/image_cache
```

驱动：`bash run.sh <cells.tsv> 0 1`（`GPU_LANE=<0|1|2>` 指定物理卡；每 cell 缓存变换 PLY，
有 `summary.json` 即跳过，半成品 cell 记 `RESUME-SKIP` 且**绝不 rm -rf**）。

## 7. 产物

- 本报告：`records/TEXTURE-BICYCLE-20261030.md`；task-7 报告已加指针行。
- 本仓库：`records/texture-voxel-20261030/`
  - `results_bicycle.jsonl`（10 行：control×2, g1, g2, g8, g10, g10b, g10c, g10d×2）
  - `summary_bicycle.txt`、`transform_stats_bicycle.txt`、`init_error_diag_alpha.json`
  - `run.sh`、`gen_cells_bicycle.py`、`resume.sh`、`summarize.py`
  - `ply_voxgrow.py`、`init_error_diag.py`（本轮版本）
- N1 原始数据：`/root/gpufree-data/outputs/bicycle-grow-20261030/`
  （`cells/<tag>/summary.json` + 官方 metrics；`ply/<tag>.ply|json`；`logs/`；`queue.log`；
  `init_error_diag_alpha.json`；`error_points_bicycle.npy`）
- 新模块：`research/init_error_diag.py`（诊断）、`research/ply_voxgrow.py`（`--dist-mode {hard,ramp2,
  traj,faronly,inneronly,holes,errdensity}`、`--under-mode {mean,le1,p10,p25,none}`、
  `--no-prune`、`--net-flat`、`--cap-scale`）。`research/ply_ops.py` 未改动。
