# 交付配方（可直接复现）

本文件是给合作者的入口：一句话说明当前配方、怎么跑、为什么是这些参数、哪些已验证/待验证。

## 0. 当前最佳已知配方与分数（2026-10-31 定稿）

> **13 例 delivery = 51.153；matched baseline = 42.323；配对增益 +8.830；13/13 场景全部提升
> （min +4.49，max +13.27，无回退）。**
> 配方：ReSplat（16 帧单次前馈，`num_refine=2`）→ FastGS 接续 **5000 步**（LR 日程匹配、
> 关 opacity reset、结构操作集中在前 1/4 步、**单 shot realloc** `frac 0.15 @ iterations//4`）
> → 官方渲染与指标。B 均值 **29.576 s**（H1 单卡串行、严格 quiet-gate）；
> 13 例 mean PSNR/SSIM/LPIPS = 26.494 / 0.8595 / 0.1839，quality 35.871 + TIME 15.282。
> 数据来源：`records/CLEAN13-20261030.md`（fresh-A hw 4 例 + reused-verified 9 例，hybrid A 已标注）。
> 旧 headline `41.371 → 49.599`（H1 `scene13r15`）**已作废**：r15 的 hw/car（67.52）与
> hw/trashbin（25.41）落在 07:16Z hw 数据损坏/重传窗口内；fresh-A 值是 **48.54 / 64.98**，
> fresh-A baseline 是 **36.90 / 55.99**。

## 1. 一句话

**冻结 ReSplat（16 帧单次前馈）→ FastGS 短程接续优化 5000 步（LR 日程匹配真实步数、
关闭不透明度重置、结构操作只允许在前 1/4 步）→ 官方渲染与指标。**
相对官方 O0 基线：**13 例平均 42.32 → 51.15（+8.83，13/13 提升、无回退）**
（`records/CLEAN13-20261030.md`，H1 单卡串行 + quiet-gate）。

> 数据沿革：旧的 `41.371 → 49.599`（`scene13r15/results.jsonl`）**已被 clean13 取代**——
> r15 的 hw/car 67.52 与 hw/trashbin 25.41 是 07:16Z hw 数据窗口的伪影；
> clean13 用 fresh-A 重测得到 48.54 / 64.98，并配了 fresh-A baseline 36.90 / 55.99。
> 其余 11 例在两批之间 reproduces within noise。更早文档里的 `42.38 → 51.35` 也已作废。

## 2. 怎么跑

```bash
python run_pipeline.py --scene <原始场景目录> --output <输出目录>
```

产出 `report.json`（PSNR/SSIM/VGG-LPIPS、B 秒数、预算、代理总分、分阶段耗时）。
依赖三个 conda 环境（见 `competition/profiles/`）：`ffsplat-enhance`（模块 A 预处理 + ReSplat）、
`organizer-fastgs`（模块 B + 官方指标）。

默认参数即交付配方，常用开关：

| 开关 | 默认 | 说明 |
|---|---|---|
| `--b-iterations` | 5000 | 模块 B 步数；实测 10k 反而总分更低（TIME 损失 > 质量收益） |
| `--b-fraction` | 0.5 | 仅作为超预算的安全阀，不是正常停止条件 |
| `--realloc-frac` | 0.15 | **单次**重分配，触发点 = `iterations // 4`（5000 步时即 1250 步）：13 例串行交付表就是这条配方（§2.1）；多 shot 已回退（同配置不可复现，见 §4） |
| `--num-refine` | 2 | ReSplat refine 轮数，>2 已饱和 |

### 2.1 13 例实测命令（不要静默改动）

`run_pipeline.py` 的默认值必须能复现下面这条模块 B 命令（`research/experiments/scene_eval_r15.sh`
delivery 臂的模块 B 参数；clean13 定稿 13 例单卡串行 delivery 均值 **51.153**，matched
baseline **42.323** → **+8.830**，见 `records/CLEAN13-20261030.md`）：

```bash
python -m research.run_sweep --case <case> --work <prepared work> --init-ply <resplat_world.ply> \
  --out <out> --iterations 5000 --snapshots "" --budget <budget> \
  --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
  --position-lr-max-steps 5000 --densify-until 1250 \
  --image-cache <work>/image_cache --realloc-frac 0.15 --realloc-at 1250
```

`run_pipeline.py` 现在按步数匹配同一日程：`densify_from = max(20, S//10)`、
`densify_until = realloc_at = max(1, S//4)`、`position_lr_max_steps = S`；在 S=5000 时
与上面的实测命令等价（5000//10=500、5000//4=1250；实测命令没显式传 `--densify-from`，
`research/train_sweep.py` 的默认值就是 500，管线显式传同一个值）。改步数做 Pareto/自适应时，
同一个日程会随 S 缩放（见 `docs/PARETO-STEPS-20261030.md` §4）。

## 3. 配方是怎么来的（每一条都有实测）

| 改动 | 效果（三例均值或指定场景） | 证据 |
|---|---|---|
| 关闭不透明度重置 | +2.1 | `docs/RESEARCH-20261030.md` §2 |
| 位置 LR 日程 = 实际步数 | hw 同 8k 步质量 32.5 → 35.4 | §3 |
| 结构操作只在前 1/4 步 | +2.2 | §4 |
| λ_dssim 0.6 / scaling_lr 0.01 | +1.8 | §5 |
| 步数 10k → 5k | TIME +3~4，质量基本持平 | §6 |
| 初始化不做锐化（用 raw ReSplat PLY） | +0.9 | §12 |
| 解码缓存（uint8 预解码） | +0.9（纯 TIME，质量不变） | §16 |
| 去掉每步 `loss.item()` 同步 | 模块 B 训练 28.22s → 24.55s（−13%） | §34 |
| SH 导出基函数修正 | +0.16（噪声内，属正确性修复） | §31 |

## 4. 已证伪、不要重复的路线

MCMC/Taming-3DGS（难例同预算 −8~10 分、易例慢 3 倍）、孔洞注入（质量 +0.3 但 B 时间翻倍、总分 −5~6）、
纯外观/冻结几何优化（q −13~−30，说明收益主要来自几何）、选图（uniform/fps/coverage 全部在噪声内）、
多子集 ReSplat（赛题约束排除）、10K/4.5K 步、粗到细分辨率、LPIPS 入损、`num_refine`>2、
激进致密化、预剪枝、位姿微调。逐条见 `docs/RESEARCH-20261030.md`。

**已证伪：统一换步数。** 13 例 × S∈{500,1000,2000,3000,5000,8000} 的 E2 扫描
（78/78 有效）+ 24 格单卡串行复测：best single uniform S 仍是 **5000**（相对现状 +0.000）；
per-scene oracle 上限只有 **+0.660**（全串行实测；串行校正 B 为 +0.646）<1 分、且在实测
单配置重跑噪声（|Δquality| 最大 0.729）量级内；每场景 argmax 方向四散
（2000/3000/5000/8000）且逐场景选参被 `AGENTS.md` 禁止。→ **不换步数、不加自适应停步**，
详见 `docs/PARETO-STEPS-20261030.md` §6/§10 与 `records/pareto-20261030/`。

**已证伪：纹理感知体素增密（texture-aware voxel grow / task-7）。** 27 格全部 exit=0，
四个变体全部丢弃：串行单卡 2 场景均值 total control **29.13** → tdm_flat **28.57（−0.56）**
→ td_plus20 **29.04（−0.09）**；bicycle quality 9.98 → 9.60（tdm_flat）/ 9.91（+20%），
**§53 的 +1.1~+1.5 在当前单 shot 配方下不复现**（那是 8-shot 时代、不裁剪的交互）；
counter 上 net-flat 三臂 −0.69/−0.61/−0.59 越过 0.5 回退线。机理：裁剪本身 −0.64（bicycle）/
−1.23（0f622ea），生长只回补 +0.3~+0.5，净负；纹理项与仅距离项差 ≤0.17（3/4 场景 ≤0.01）；
32³ 欠高斯均值闸门选中 87–96% 有高斯体素，等于没筛；>0.5R 远区先验只覆盖 5.7–11.2%，
在环形 billboard 退化到 94.6%；模块 B 本来就会重新致密化，多出的 init 基元只留下 TIME 成本。
→ `--init-ply` 继续用**原始** `resplat_world.ply`；详见 `docs/RESEARCH-20261030.md` §56 与
`records/TEXTURE-VOXEL-20261030.md`。

**已回退：多次容量重分配（multi-shot realloc）。** 8 次重分配在质量上曾看到 +5.1/+3.3/+0.9，
但同一配置的顺序复跑出现 **6.55 vs 10.86** 质量分（mip/bicycle，差 4.3 分）和 28.4 vs 44.1 s 的
B 时间漂移，无法复现；并发还会把 B 墙钟抬 60~80%。因此交付配方固定为**单 shot**
（`realloc_frac 0.15, realloc_at = iterations//4`），判据重写与复测见
`research/advisory/HANDOFF_MULTISHOT_REALLOC_20261030.md`。

## 5. 关键约束（不要违反）

1. **最多 16 帧、单次前馈**，ReSplat 权重与网络冻结，`num_refine=2`；
2. 预算 = 同卡 FastGS **随机初始化 30000 步**的实测秒数（dl3dv 94.29 / hw 99.49 / mip 92.57 秒）；
3. TIME 项只计模块 B（已与用户确认），A/ReSplat 不计时；
4. 计时必须**单卡串行**：并发会让 B 秒数波动 ±45%，同场景甚至出现过 31.4/36.1/54.2 秒的读数；
   本次 E2 的 24 格串行/并发配对实测中位只高 +3.2%（0.859–0.994），但冷启动格可到 +16%，
   所以交付计时仍按单卡串行口径；
5. 不要删除赛方原始数据集目录（曾有脚本误删，需重新上传）。

## 6. 已知边界

- 三个默认例口径 62.01，但 13 例只有 51.15（clean13）：难例（dl3dv/0b5635 31.1、mip/bicycle 25.31）
  的前馈重建本身质量差（LPIPS 0.24~0.31）。难例在 500→2000→5000 步上质量仍在单调上升
  （0b5635 质量 5.19→10.67→17.06，见 `docs/PARETO-STEPS-20261030.md` §3），所以「固定 5000 步」
  对难例可能仍欠训练、对易例可能过训练——这正是步数 Pareto 扫描要分场景裁决的问题；
- 输入图不是模糊问题：低分场景的 Laplacian 方差比高分场景高 3~4 倍（§25）；
- **75 分不可达**：三例把模块 B 时间当 0 的上界约 65，13 例集合更难。
- **hw 组分数有历史 A 批次问题（已定稿解决，不是 run-to-run 随机）**：模块 A 本身可复现
  （a-poses P0：3 次全新 A 的 registered 数一致，gauge-invariant 相对旋转中位 0.023°/0.037°、
  平移 0.017°/0.035°、Umeyama 0.02°/0.03%；输入数据与 canonical 副本 md5/清单一致）。
  旧的 `scene13r15` 批次（07:16 UTC）hw/car、hw/trashbin 落在数据损坏/重传窗口内：
  其 67.52 / 25.41 不可用；clean13 用 fresh-A（16:25–16:46 UTC）重测得 delivery
  **48.54 / 64.98**、matched baseline **36.90 / 55.99**，与 N1/E2 的独立 draw
  （car 47.95、trashbin 65.94）一致。**当前定稿分数用 clean13 的 42.32 → 51.15。**
  稳定性矩阵见 `records/POSE-REPLACEMENT-20261030.md`，干净批次见
  `records/CLEAN13-20261030.md`。
- **步数不是当前杠杆**（E2 结论）：13 例 × S∈{500..8000} 扫描里 best uniform S 仍是 5000
  （+0.000），per-scene oracle 上限 +0.66（全串行实测）/ +0.79（raw）；见
  `docs/PARETO-STEPS-20261030.md` §6/§10。

## 7. 可选 M1 门控（默认关闭，2026-10-31 用户批准接入）

**做什么**：在模块 B **之前**，用 `research/prune_gate.py` 从 `resplat_world.ply` 渲染
**8 个固定训练视角**、算与训练 GT 的 VGG-LPIPS（不碰测试图）。若该 train-LPIPS >
`--mid-prune-tau`（校准值 **0.48**），则给模块 B 追加 `--mid_prune_at iterations//4
--mid_prune_keep 0.6`（M1 中途按"训练视角 2D 梯度范数重要性"物理剪枝 40%，再以同一日程
跑完 5000 步）；否则不加任何参数、与当前交付配方完全一致。

**预期增益**：13 例校准中只有 `mipnerf360/bicycle` 触发（train-init LPIPS 0.524 vs 次高
0.446）。bicycle 上 M1 keep=0.6 实测 **Δtotal +0.84**，因此 13 例平均
**+0.84/13 ≈ +0.065** —— 小但确定为正。Spearman(train-init LPIPS, delivered test LPIPS)
= **0.973**（p<1e-4），门控的代理关系有数据支撑。

**怎么开**：`python run_pipeline.py --scene ... --output ... --mid-prune auto`
（可调 `--mid-prune-tau` / `--mid-prune-keep`）。门控 JSON 与耗时写入 `report.json`
（`mid_prune`、`stage_seconds.prune_gate`；门控在 B 计时之外，因此是"免费"的）。

**为什么默认 off**：① 收益只由 bicycle 单场景承载（13 例平均 +0.065 与 B 的 ±0.5 s
run-to-run 噪声同量级）；② 若评测集出现 train-init LPIPS >0.48 但交付 LPIPS 仍 <0.25 的
场景，误触发会损失约 1–2 分；③ 默认 off 保证已测定的 51.153 交付配方与复现命令不变。
