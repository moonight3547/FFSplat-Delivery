# 纹理感知体素增密实验（2026-10-30/31，N1）

**结论：四个变体全部不采纳（drop）。** 在净数量不变（net-flat）的前提下，
"低不透明度裁剪 → 在欠高斯体素里按 纹理×距离 重要性分裂 +20%" 在 4 个场景上
**质量平均 −0.42 ~ −0.53 分**，串行单卡锚点（2 个难例）上总分 **−0.56**；
不裁剪的 +20%（净 +16.1% 数量、模块 B 时间变贵）质量 **+0.23** 但 B **+1.3 秒**，
总分 **−0.09**，收益没有兑现。纹理项与"仅距离"项的结果差 ≤0.05 分（噪声级），
说明在本文的这个实现里**纹理加权没有带来额外信息**。

---

## 1. 做了什么（严格按用户/Lead 收紧后的要求）

用户原话（2026-10-31 01:07）：*"体素的权重可以参考 pi3_sample_points.py，先 opacity prune，
再增加 20% 的高斯分裂补充在欠高斯的体素区域，距离 scene_center 越远应该有更多的权重……
也可以测试给距离 scene_center 超过一半 scene_radius 的 2 倍权重……"*
Lead 追加：**时间在模块 B，不要浪费；net-flat 是主臂；体素用 32³；实现要 seconds 级。**

新模块 `research/ply_voxgrow.py`（`research/ply_ops.py` 未改动，只 import `load/save/sigmoid/logit`）：

1. **opacity prune**
   - 普通臂：`sigmoid(opacity) < --prune-min-opacity`（0.01，沿用 `ply_ops.op_prune` 的浮点阈值）；
   - **`--net-flat`（主臂）**：按不透明度从低到高裁掉 `N - round(N/(1+grow_frac))` 个
     （+20% 时 = 16.7%），使得增密后**总数回到原始值** → 模块 B 的 TIME 不变。
     实测的等效不透明度上限：bicycle 0.052、0f622ea 0.039、billboard 0.053、counter 0.017。
2. **+20% 生长**：把新增的 `round(0.2 × N_after_prune)` 个高斯**分裂**到欠高斯体素。
   **欠高斯体素定义（可复现）**：在 0.5–99.5% 分位包围盒上铺 `--voxels`³ 网格，统计每个
   *有高斯* 体素的计数 `c_v`；令 `mean_occ = mean(c_v)`（只统计有高斯的体素）。
   `c_v < mean_occ` 的体素即"欠高斯"，其 *deficit* = `ceil(mean_occ) - c_v`；
   生长预算按重要性分配、单个体素上限 = deficit（预算装不下时上限按 2×/4×… 放宽）。
   每个子高斯 = 从该体素内随机取一个父高斯复制，scale ÷1.6，位置抖动 0.3σ（与
   `op_texresample` 一致的 FastGS 分裂语义）。
3. **重要性（pi3 结构的 PLY 版）**
   ```
   importance = (minimum_weight + texture_weight · texture^gamma) · distance_prior
   texture     = 体素内平均逆尺度（小基元=高细节），按体素经验分位映射到 [0,1]
                 → 直接沿用 pi3 的 minimum_weight=0.05, texture_weight=4.0, gamma=1.25
   distance_prior = 1.0 ；若体素质心到 scene_center 的距离 > far_frac·scene_radius 则 = 2.0
   scene_center   = 0.5–99.5% 分位包围盒中心；scene_radius = 该包围盒对角线的一半
                    （= 鲁棒包围球半径）
   ```
   PLY 中没有 confidence / 多视角 support，这两项按任务说明省略（已在代码注释中写明）。
   pi3 的"中心先验"被**距离优先取代**（用户要求的方向相反：越远权重越大）。
4. **V2 合并**（`grow_texdist_merge`）：裁剪前先把每个被裁高斯的属性并入**最近的幸存邻居**
   ——不透明度相加并 clamp 到 0.999（logit 存回）、f_dc/f_rest 按不透明度加权平均、
   scale 逐轴取 max、位置/旋转保留幸存者。最近邻用 `scipy.cKDTree`（有 numpy 兜底实现）。

**变体**：`grow_texdist`（纹理×距离）、`grow_texdist_merge`（同上+合并）、
`grow_distonly`（只有距离）、`prune`（只裁剪，诊断臂：把"裁剪的损失"和"生长的收益"分开）。
按 Lead 要求**去掉了 `grow_texonly`**（`grow_distonly` vs `grow_texdist` 已能分离纹理贡献）。

**效率**：变换全程向量化（`np.bincount` 统计 + 最大余数分配 + `np.repeat` 一次生成子高斯 +
`reduceat` 合并），**每个 40–52 万高斯 PLY 0.4–2.0 秒**（合并臂最慢 2.0 s），输出 PLY 缓存，
重跑 0 CPU 成本。不渲染、不用测试图像。

## 2. 实验矩阵与命令

场景（4）：`mipnerf360/bicycle`、`dl3dv/0f622ea`（难例）；`hw/billboard`、`mipnerf360/counter`（易例对照）。
臂：`control`、`td_flat`、`tdm_flat`、`do_flat`（16 cell @ 3 lanes）；
外加 `td_plus20`（bicycle+0f622ea，给 +20% 定价）、`td_flat64`（bicycle 64³ 敏感度）、
`prune_only`（bicycle+0f622ea，分解裁剪损失）→ 并行 21 cell；
再串行单卡（其余两卡空闲）跑 `control / tdm_flat / td_plus20` × bicycle+0f622ea = 6 cell。
共 **27 cell，全部 exit=0**，raw 数据在 `/root/gpufree-data/outputs/texvox-20261030/`。

模块 B 交付配方**一字未改**，只换 init PLY：

```bash
# 变换（fastgs env, PYTHONPATH=repo root）
python -m research.ply_voxgrow --in resplat_world.ply --out grown.ply \
    --op grow_texdist --net-flat --voxels 32 --grow-frac 0.2 \
    --far-frac 0.5 --far-weight 2.0 --seed 0 --summary-out grown.json

# 模块 B（3 lanes：competition/profiles/gpufree-{0,1,2}.json）
python -m research.run_sweep --profile competition/profiles/gpufree-<lane>.json \
  --case <case> --work <work> --init-ply <transformed PLY> --out <out> \
  --iterations 5000 --snapshots "" --budget <budget> \
  --position-lr-max-steps 5000 --opacity-reset-interval 0 --lambda-dssim 0.6 \
  --scaling-lr 0.01 --densify-from 500 --densify-until 1250 \
  --realloc-frac 0.15 --realloc-at 1250 --image-cache <work>/image_cache
```

驱动：`records/texture-voxel-20261030/run.sh <cells.tsv> <LANE> <NLANES>`（可续跑、缓存变换 PLY、
每 cell 记录 `start/exit` 到 queue.log、跑完删 model 目录省盘）。

## 3. 结果（并行 3 lanes）

质量 q = PSNR/SSIM/LPIPS 的官方线性代理分；B = 模块 B 秒数（load+init+train+0.5）；
TIME = 20·min(1,(1−B/budget)/0.9)；total = q + TIME。变换数量 `n_in→n_out`，`n_train` = 模块 B 训练结束后的高斯数。

| 场景 | 臂 | n_in→n_out | n_train | PSNR | SSIM | LPIPS | q | B(s) | TIME | total | Δq vs ctl |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | control | — | 1 001 646 | 24.54 | 0.730 | 0.298 | 10.01 | 32.1 | 14.70 | 24.71 | — |
| bicycle | td_flat | 520 980→520 980 | 996 498 | 24.38 | 0.722 | 0.309 | 9.27 | 27.5 | 15.79 | 25.06 | **−0.73** |
| bicycle | tdm_flat | 520 980→520 980 | 1 028 564 | 24.47 | 0.727 | 0.304 | 9.69 | 27.7 | 15.75 | 25.44 | **−0.31** |
| bicycle | do_flat | 520 980→520 980 | 996 984 | 24.39 | 0.722 | 0.309 | 9.28 | 27.4 | 15.82 | 25.11 | −0.72 |
| bicycle | prune_only | 520 980→434 150 | 893 423 | 24.41 | 0.723 | 0.308 | 9.37 | 23.9 | 16.63 | 26.00 | −0.64 |
| bicycle | td_plus20 | 520 980→604 639 | 1 103 398 | 24.50 | 0.728 | 0.300 | 9.85 | 30.4 | 15.12 | 24.97 | −0.15 |
| bicycle | td_flat64 | 520 980→520 980 | 1 003 360 | 24.44 | 0.724 | 0.307 | 9.45 | 26.7 | 15.97 | 25.42 | −0.55 |
| 0f622ea | control | — | 880 387 | 22.57 | 0.795 | 0.230 | 17.76 | 38.4 | 13.18 | 30.93 | — |
| 0f622ea | td_flat | 476 635→476 635 | 884 045 | 22.41 | 0.792 | 0.232 | 16.98 | 35.6 | 13.84 | 30.81 | **−0.78** |
| 0f622ea | tdm_flat | 476 635→476 635 | 891 181 | 22.46 | 0.793 | 0.232 | 16.94 | 35.5 | 13.85 | 30.79 | **−0.82** |
| 0f622ea | do_flat | 476 635→476 635 | 887 516 | 22.44 | 0.792 | 0.232 | 16.98 | 31.6 | 14.78 | 31.76 | −0.78 |
| 0f622ea | prune_only | 476 635→397 196 | 813 201 | 22.40 | 0.791 | 0.233 | 16.53 | 31.2 | 14.86 | 31.39 | −1.23 |
| 0f622ea | td_plus20 | 476 635→550 152 | 956 763 | 22.56 | 0.797 | 0.227 | 18.38 | 38.4 | 13.17 | 31.55 | +0.63 |
| billboard | control | — | 574 269 | 22.72 | 0.807 | 0.140 | 36.76 | 28.0 | 15.97 | 52.73 | — |
| billboard | td_flat | 387 019→387 019 | 573 892 | 22.72 | 0.807 | 0.139 | 36.89 | 23.9 | 16.88 | 53.77 | +0.13 |
| billboard | tdm_flat | 387 019→387 019 | 580 253 | 22.69 | 0.807 | 0.139 | 36.81 | 23.9 | 16.88 | 53.69 | +0.05 |
| billboard | do_flat | 387 019→387 019 | 575 468 | 22.71 | 0.807 | 0.140 | 36.72 | 23.2 | 17.03 | 53.75 | −0.04 |
| counter | control | — | 619 995 | 27.89 | 0.901 | 0.181 | 40.24 | 28.8 | 15.32 | 55.55 | — |
| counter | td_flat | 488 831→488 831 | 647 414 | 27.70 | 0.901 | 0.183 | 39.55 | 28.2 | 15.45 | 55.00 | **−0.69** |
| counter | tdm_flat | 488 831→488 831 | 650 099 | 27.83 | 0.900 | 0.183 | 39.62 | 27.5 | 15.62 | 55.25 | **−0.61** |
| counter | do_flat | 488 831→488 831 | 647 437 | 27.70 | 0.901 | 0.183 | 39.65 | 30.0 | 15.01 | 54.66 | −0.59 |

**并行 4 场景均值**（B 受 3 卡并发影响，只作参考）：

| 臂 | q | B(s) | TIME | total |
|---|---:|---:|---:|---:|
| control | **26.19** | 31.8 | 14.79 | 40.98 |
| td_flat | 25.67 | 28.8 | 15.49 | 41.16 |
| tdm_flat | 25.77 | 28.7 | 15.52 | 41.29 |
| do_flat | 25.66 | 28.1 | 15.66 | 41.32 |

> **并行 B 有会话漂移陷阱**：control 4 个 cell 最先跑（B 均值 31.8 s），flat 臂后跑（28.1–28.8 s），
> 于是 flat 臂的 total 看起来 +0.2~+0.3。**这是漂移，不是臂效应** —— 同一次会话内同一场景
> 同一计数下 B 也能差 4 s（0f622ea td_flat 35.6 vs do_flat 31.6）。裁决必须看下面的串行锚点。

## 4. 串行单卡锚点（其余 2 卡空闲，B/TIME 干净）

| 场景 | 臂 | q | B(s) | TIME | total | Δq | ΔB | ΔTIME | Δtotal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | control | 9.98 | 27.2 | 15.85 | 25.83 | — | — | — | — |
| bicycle | tdm_flat | 9.60 | 26.8 | 15.96 | 25.56 | −0.38 | −0.5 | +0.11 | **−0.28** |
| bicycle | td_plus20 | 9.91 | 28.4 | 15.57 | 25.48 | −0.07 | **+1.2** | −0.28 | **−0.35** |
| 0f622ea | control | 17.95 | 32.9 | 14.48 | 32.43 | — | — | — | — |
| 0f622ea | tdm_flat | 17.07 | 32.7 | 14.51 | 31.58 | −0.88 | −0.1 | +0.03 | **−0.85** |
| 0f622ea | td_plus20 | 18.47 | 34.4 | 14.13 | 32.59 | +0.52 | **+1.5** | −0.35 | **+0.17** |
| **2 场景均值** | control | 13.96 | 30.0 | 15.16 | **29.13** | | | | |
| | tdm_flat | 13.33 | 29.8 | 15.23 | **28.57** | −0.63 | −0.3 | +0.07 | **−0.56** |
| | td_plus20 | 14.19 | 31.4 | 14.85 | **29.04** | +0.23 | +1.3 | −0.32 | **−0.09** |

**质量可重复性**（同配置 parallel vs serial）：bicycle control 10.01 vs 9.98（+0.03）、
0f622ea control 17.76 vs 17.95（+0.19）⇒ 单跑质量噪声约 ±0.2 分。上表的 −0.63 与 +0.23 都是真信号。

## 5. 怎么读这些数（结论的四条证据）

1. **纹理项没有额外贡献**：`td_flat` 与 `do_flat`（只距离、无纹理）在 4 个场景上
   q 差 = +0.01 / 0.00 / +0.17 / −0.10（|Δ| ≤ 0.17，其中 3 个 ≤0.01）。
   按 pi3 的 `texture_weight=4.0` 加权的纹理代理在这个 PLY 上被距离项完全淹没。
2. **"欠高斯"闸门在 32³ 下几乎不筛选**：mean_occ = 68.4/66.1/42.1/27.2，
   而低于均值的体素占**有高斯体素的 87%–96%**（bicycle 5526/6343，0f622ea 13269/14615，
   billboard 4672/4877，counter 8748/9668）。所以"补充到欠高斯区域"实际退化成
   "全场景按重要性重分配"。64³（bicycle，16.9 均值）也是 87%。这是定义层面的弱点。
3. **距离先验的杠杆很小而且在 ring 场景退化**：
   `far_frac·scene_radius` 阈值下，落在"远区"的**高斯占比**是 bicycle 9.7%、0f622ea 5.7%、
   counter 11.2%，而 billboard **94.6%**（点云是环形，离包围盒中心距离几乎恒定）。
   也就是说用户设想的"背景/远处"在 3 个场景里只是最外 5–11% 的稀疏尾巴，
   在 billboard 里是整个场景（先验退化为常数 2×）。
4. **裁剪本身是主要损失，生长只能补回一小部分**（`prune_only` 诊断臂，质量为准）：
   - bicycle：control 10.01 → prune_only 9.37（**−0.64**）→ td_flat 9.27（生长再 −0.10）
     → tdm_flat 9.69（合并生长比纯裁剪 **+0.32**）；
   - 0f622ea：control 17.76 → prune_only 16.53（**−1.23**）→ td_flat 16.98（生长 **+0.45**）
     → tdm_flat 16.94（+0.41）。
   生长机制**确实有正向作用**（+0.3~+0.5），"先合并再丢弃"也比直接丢弃好一点，
   但都填不上"裁掉 16.7% 最低不透明度"挖的坑。这与 §53（128³、不裁剪、常量数量重分配）
   的方向不矛盾：那里的收益来自"重分配"，不是"先裁剪"。
5. **+20% 的数量代价真实存在**：净 +16.1% 初始基元（520 980→604 639 / 476 635→550 152）
   传到训练结束仍是 +10.2% / +8.7%（1 001 646→1 103 398 / 880 387→956 763），
   代价是 B **+1.2 s / +1.5 s**（+4.4% / +4.6%），TIME **−0.28 / −0.35**；
   质量只 +0.23（2 场景均值，其中 bicycle −0.07），**总分 −0.09 → 不划算**。
   若换到 TIME 权重更高的场景（B 更小、预算更小），这个亏损只会更大。

## 6. 采纳 / 丢弃

| 变体 | 判定 | 理由（数字） |
|---|---|---|
| `control`（原始 ReSplat PLY） | **保留**（基线） | — |
| `td_flat`（纹理×距离，net-flat） | **丢弃** | 4 场景 Δq −0.53；串行 2 场景 Δtotal −0.56（tdm 版） |
| `tdm_flat`（+ 裁剪前合并到最近邻） | **丢弃** | 比 td_flat 好 +0.10（4 场景均值）但仍是 −0.42；串行 Δtotal −0.56 |
| `do_flat`（只有距离先验） | **丢弃** | Δq −0.53，与 td_flat 无差别 → 纹理项无用 |
| `td_plus20`（+20% 生长，净 +16.1%） | **丢弃** | Δq +0.23 但 dB +1.3 s → Δtotal −0.09；且未在 counter/billboard 上验证 |
| `td_flat64`（64³ 敏感度，bicycle） | **丢弃** | Δq −0.55，比 32³（−0.73）略好但依然为负 |
| `prune_only`（只裁剪 16.7%） | 仅诊断 | 证明损失主要来自裁剪（−0.64 / −1.23） |

**采纳**：无。交付配方**保持不变**（`--init-ply` 继续用原始 `resplat_world.ply`）。

也满足"易例不得回退 >0.5"这条：`counter` 上 net-flat 三臂分别 −0.69 / −0.61 / −0.59，
**已经越过 0.5 的回退线**——这是拒绝它们的独立理由；`billboard` 基本持平（−0.04~+0.13）。

## 7. 后续（如果还要继续这条线）

按性价比排序，都没有在本次跑（时间/资源原因，未验证）：
1. **只生长、不裁剪的同数量重分配**：把 §53 的 texresample 语义（目标计数 ∝ 纹理×当前计数、
   常量总数）在 32³/64³ 上、用当前单次 realloc 配方重测——这是唯一在旧配方下有正证据的形态。
2. **更细、更有判别力的欠高斯闸门**：用 `count_v < 25% 分位` 或 `count_v <= 1` 而不是均值闸门
   （本次均值闸门选中 87–96% 的体素，等于没筛）。
3. **只看"远区"是否缺覆盖**：先量化远区（>0.5R）的覆盖/不透明度统计，再决定要不要给 2× 权重；
   对 billboard 这类环形点云需要按"到相机轨迹的距离"而不是"到包围盒中心的距离"。
4. 合并思路（V2）本身值得单独保留：它在 bicycle 上比"直接裁剪"多 +0.32，
   如果未来必须裁剪（比如强制降基元数），**先合并再丢**是更好的裁剪方式。

## 7.1 后续研究（已执行）

- **2026-10-31 续篇（bicycle-only、grow-only、无裁剪；距离/空洞/误差密度三种权重）**：见
  `records/TEXTURE-BICYCLE-20261030.md`。**结论全负、无变体采纳**。α 修正版训练视角诊断：
  原始 init 平均 α=0.9964、α<0.5 像素 **0.0000**（无空洞）、>0.5R 的误差占比 6.52% **低于**
  高斯占比 8.74%（误差不在外围）、逐像素误差内圈最高（0.0278/0.0282 vs 中环 0.0147）。
  四种空间权重质量 −0.04~+0.13（噪声带内），每个 +20% 却要 +1.7~3.1 s 的 B，
  全部 Δtotal 为负（−0.34~−0.66）。唯一有信号的方向是"误差密度生长"（+0.13 质量），
  但时间成本仍吃掉全部收益。

## 8. 产物

- 本仓库：`records/TEXTURE-VOXEL-20261030.md`（本文）、
  `records/texture-voxel-20261030/{results.jsonl, summary.txt, summary.md, run.sh, gen_cells.py,
  summarize.py, ply_voxgrow.py, transform_stats.json}`
- N1 原始数据：`/root/gpufree-data/outputs/texvox-20261030/`
  （`cells/<tag>/summary.json` + `eval/` 官方 metrics、`ply/<tag>.ply|json` 变换产物与统计、
  `logs/`、`queue.log`、`results.jsonl`）
- 新模块源码：`research/ply_voxgrow.py`（N1 仓库同文件）。

## 9. 复现

```bash
# N1
cd /root/gpufree-data/outputs/texvox-20261030
/opt/conda/envs/fastgs/bin/python gen_cells.py main   > cells.tsv    # 21 cell
/opt/conda/envs/fastgs/bin/python gen_cells.py serial > serial.tsv   # 6 cell
for L in 0 1 2; do setsid nohup bash run.sh cells.tsv $L 3 > lane$L.out 2>&1 & done
setsid nohup bash run.sh serial.tsv 0 1 > lane_serial.out 2>&1 &     # 单卡锚点
/opt/conda/envs/fastgs/bin/python summarize.py                       # -> results.jsonl + 表
```

**注意**：本次三卡并行 sweep 01:15–01:25、串行锚点 01:25–01:35（当时 N1 三卡空闲）。
01:37 起 N1 上出现了另一个任务（`stagedloss-20261030`，占用 GPU2），
之后的 N1 计时必须重新做空闲确认。
