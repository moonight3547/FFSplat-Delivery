# 状态继承文档（H1 / N1 + 当前实验）

> 生成时间：2026-10-30 23:00（Asia/Shanghai）
> 来源会话：`session-b90880e3-38e0-4559-bd02-a2eccc103404`（标题「阅读FFSplat项目并测定baseline」，已结束）
> 本会话（新对话）从这里继续。本文所有节点状态均为 **2026-10-30 22:47–23:00 的 SSH 实连采集**，不是转述旧日志。

---

## 0. 本会话的三条新指令（用户，2026-10-30，最高优先级）

| # | 指令 | 对实验计划的直接后果 |
|---|---|---|
| 1 | **评测分辨率是原始分辨率**，这是赛题要求，已明确确认 | **旧会话提议的「降分辨率换 LPIPS」实验（960→832/768/640/512）作废**。不得再把降分辨率当作提分杠杆；评测口径以原始分辨率对齐。 |
| 2 | **步数的 Pareto 前沿可以测试** | 批准执行：模块 B 步数 S∈{500,1000,2000,3000,5000,8000} × 13 例，找每例最优点，见 §4。 |
| 3 | **H1、N1 的服务器信息与当前实验信息需要明确继承，在本对话继续安排实验** | 即本文。 |

旧会话最后提出的三条「大颗粒」实验：① 分辨率 → **被 #1 否决**；② 步数 Pareto → **被 #2 批准**；③ 按场景自适应 B 预算 → 作为 ② 的下游推论保留（用 train-only 信号，不碰测试集）。

---

## 1. 节点总览（本会话接手 H1 / N1；H2/H3/N2 已交 codex，不重复占用）

| 代号 | SSH | 卡 | 角色 | 22:50 实测状态 |
|---|---|---|---|---|
| **H1** | `ffh1`（本地别名）→ `ssh -p 22304 root@cpod-1wqzm4msan2q-s3.podtcp.compshare.cn` | 1 × 4090 **48G** | **准备机**：唯一同时具备 COLMAP + ReSplat(`ffsplat-enhance`) + FastGS 的节点；产出 A/ReSplat 世界系 PLY 与预算 | 22:50 在线、GPU 0%；**已换新 pod：根盘 200G**（旧记录 50G/1.1G 作废），剩 132G→23:20 的 124G；SSH 握手仍间歇 `Connection closed` |
| **N1** | `ssh -p 30130 root@183.147.142.40` | 3 × 4090 **24G** | **评测机**：13 例 B-only + 官方指标并行 | 22:50 在线、3 卡 0%；根盘 30G **99% 满（312M→309M）**，`/root/gpufree-data` 246G / 剩 142G→23:20 的 ~118G；`outputs` 已软链到 gpufree |

- 认证：本机 `~/.ssh/id_ed25519` 对两台免密；H1→N1 直连用 H1 上的 `~/.ssh/ton1`（实测可用），因此**大文件从 H1 直传 N1，不绕本机**。
- 本机新增工具：`scripts/remote.sh <h1|n1> '<cmd>'` —— 对 H1 握手失败自动重试（`ControlPath=/tmp/.sshcm/%C`，注意不能放 `~/.ssh`，沙箱禁止）。
- 旧文档 `GPU_NODES_20261030.md` 里的 H1 磁盘/HEAD 已过时，以本文为准。

---

## 2. 两台的代码与环境现状（重要：尚未与本地 main 对齐）

| 项 | H1 | N1 |
|---|---|---|
| 仓库路径 | `/workspace/organizer-projects/FFSplat-Hackathon` | `/root/projects/FFSplat-Hackathon` |
| HEAD | `cd355cb`（工作树：3 个已跟踪文件已改 + 13 个未跟踪文件；train_sweep +264 行，多 shot realloc 实现） | `c3c1ee5`（工作树：8 个已跟踪文件改动[6M+2D] + 11 个未跟踪文件，其中 2 个由本轮任务新增） |
| conda envs | `ffsplat-enhance`(ReSplat/py3.12/torch2.7/gsplat1.5.3)、`organizer-fastgs`、`organizer-toolkit124`、`py312` | 仅 `fastgs`（ReSplat env 已被删除以腾空间） |
| 数据 | `/workspace/organizer-assets/datasets` 13 例齐全 | 无原始数据，只有已准备的 work + PLY |
| 权重 | `/workspace/organizer-assets/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth` | 不需要 |

- 本地 Mac 仓库（`FFSplat-Hackathon/`）main = **`ad90ed4`**，已 push 到 `git@github.com:moonight3547/FFSplat-Hackathon.git`。
- **两台的 working tree 都未与 `ad90ed4` 对齐**。新实验开始前必须先决定用哪份代码；本次 Pareto 扫描按「交付配方实测命令」复刻（见 §4），参数层面不依赖多 shot 的那些未提交改动（`--realloc-frac 0.15` 单次 shot 在两边实现里行为一致）。

---

## 3. 当前实验信息（**快照 22:50 时两台全部空闲**；23:15 起 H1 跑 LPIPS 筛查、N1 跑 13 例 Pareto 扫描，实时进度见 §4）

### 3.1 已完成的 13 例配对结果（H1 单卡串行，最高价值基线表）

文件：H1 `outputs/scene13r15/results.jsonl`（13 行，脚本 `research/experiments/scene_eval_r15.sh`）

| case | baseline 总分 | delivery 总分 | Δ | q: base→del | B(s): base→del |
|---|---:|---:|---:|---:|---:|
| dl3dv/001dccbc… | 66.64 | 78.99 | +12.35 | 56.4→64.6 | 51.0→33.2 |
| dl3dv/0b5635… | 25.61 | 31.13 | +5.53 | 15.1→17.2 | 49.6→35.2 |
| dl3dv/0f622ea… | 22.15 | 31.86 | +9.71 | 12.8→18.2 | 54.8→36.5 |
| dl3dv/2dad2d… | 36.49 | 49.60 | +13.11 | 25.6→35.3 | 48.0→33.8 |
| dl3dv/3e4dbe… | 26.03 | 35.29 | +9.26 | 15.7→21.1 | 50.5→34.2 |
| dl3dv/9a4516… | 52.13 | 59.56 | +7.43 | 40.6→44.4 | 45.4→29.9 |
| hw/billboard | 45.56 | 53.38 | +7.82 | 32.5→36.5 | 40.9→23.9 |
| hw/basketball | 58.96 | 67.55 | +8.59 | 45.5→50.5 | 37.3→22.1 |
| hw/car | 58.24 | 67.52 | +9.28 | 44.9→50.5 | 38.0→22.1 |
| hw/trashbin | 21.91 | 25.41 | +3.50 | 8.1→9.0 | 36.0→24.9 |
| mipnerf360/bicycle | 20.82 | 25.32 | +4.50 | 10.8→9.9 | 52.1→29.2 |
| mipnerf360/bonsai | 56.53 | 63.93 | +7.40 | 44.6→48.3 | 44.0→28.2 |
| mipnerf360/counter | 46.74 | 55.23 | +8.49 | 35.4→40.3 | 45.5→30.3 |
| **均值** | **41.371** | **49.599** | **+8.228** | | |

> 另一组数字 `42.38 → 51.35` 出自 `docs/RESEARCH-20261030.md` **§24**：其 O0 对照跑的是
> `position_lr_max_steps 30000`（未随 10k 步匹配），与本表 §3.2 的匹配日程口径不同，
> 两者差在 RESEARCH 第 728 行有记录（`42.38 vs 41.37`）。**以本表（落盘 JSONL）为准**，`51.35` 不再引用。
> 注意 `mip/bicycle` 质量分 10.8→9.9，是 13 例中唯一的 delivery 质量回退（但 TIME 回补，总分仍 +4.5）。

### 3.2 交付配方（实测命令，来自 `scene_eval_r15.sh` + `run.log` 落盘）

模块 B（`research.run_sweep` 调用，**缺任一 flag 都不可复现 TIME/总分**）：

```bash
python -m research.run_sweep --case <case> --work <A-work> --init-ply <resplat_world.ply> \
  --out <out>/del --iterations 5000 --snapshots "" --budget <budget> \
  --opacity-reset-interval 0 --lambda-dssim 0.6 --scaling-lr 0.01 \
  --position-lr-max-steps 5000 --densify-until 1250 \
  --image-cache <cache> --realloc-frac 0.15 --realloc-at 1250
```
- `--seed` 默认 `20261015`（未显式传，值即此）；
- `--densify-from` **未传** = FastGS 默认 500；
- 展开后的 `research.train_sweep` 逐字命令见 `outputs/scene13r15/run.log`（含 `--sh_degree 3 -r 1 --eval --opt_step_schedule --inject_holes 0 --inject_at 800`）。
- 模块 A：`module_a/preprocess.py --num_context 16 --llffhold 8 --max_resolution 960`；
- ReSplat：`research.resplat_variant --num_refine 2 --device cuda:0 --fix-sh-basis --checkpoint <ckpt>`；
- baseline O0：`--iterations 10000 --position-lr-max-steps 10000 --densify-until 15000 --opacity-reset-interval 3000 --realloc_frac 0.0 --realloc_at 1250`。

### 3.3 已有的步数曲线（H1 `outputs/ocurve/`，3 个难例 × 500/2000/5000）

| case | S=500 (q / B / total) | S=2000 | S=5000 |
|---|---|---|---|
| dl3dv/0b5635 | 5.19 / 12.48 / 24.47 | 10.67 / 18.49 / 28.54 | 17.06 / 36.05 / **30.79** |
| dl3dv/0f622ea | 3.78 / 10.33 / 23.57 | 6.02 / 17.58 / 24.10 | 17.82 / 36.57 / **31.42** |
| dl3dv/3e4dbe | 5.41 / 9.85 / 25.31 | 缺 | 缺 |

> 难例上步数越多总分越高（质量增长压过 TIME 损失），说明「统一砍到很少步数」是错的；关键在于**易例是否早就饱和**——这正是 §4 要测的。

### 3.4 其它已确认/已证伪（继承，不重复踩）

- **已证伪**：MCMC/Taming-3DGS（难例同预算 −8~10 分）、孔洞注入（质量 +0.3 但 B 翻倍）、纯外观/冻结几何（−13~−30）、A 侧 fps/coverage 选图（噪声内）、多子集 ReSplat（赛题排除）、10K/4.5K 步、粗到细分辨率、LPIPS 入损、`num_refine>2`、激进致密化、预剪枝、位姿微调、经典 3DGS 当 B、渲染端抗锯齿（膨胀高斯单调变差）。
- **确认收益**：去每步 `loss.item()` 同步（B 28.22→24.55s）、SH 导出基函数修正（+0.16，正确性修复）、uint8 解码缓存（+0.9 TIME）、结构操作只在前 1/4 步、position LR 日程=真实步数、关 opacity reset。
- **多 shot 重分配 = 高方差**：同配置两次 6.55 vs 10.86（质量分；证据 `research/advisory/HANDOFF_MULTISHOT_REALLOC_20261030.md:23–36`，同配置/同 GPU/顺序执行/同 init，差 4.3 质量分与 15.7 秒），已交 codex 调查。交付配方保持**单 shot**（`realloc_frac 0.15, realloc_at = iterations//4`）。✅ 本地 `run_pipeline.py` 默认值**已修复**为单 shot（此前是 8 次），`DELIVERY.md` 同步更正。
- **测量注意**：并发会抬高 B 墙钟（同配置 28~44s，±45%）；**TIME 结论必须单卡串行复测**。

---

## 4. 本对话的实验安排（进行中）

| 编号 | 实验 | 责任 | 状态 |
|---|---|---|---|
| E1 | H1→N1 补齐 6 个缺失场景 + 全 13 例 image_cache + 13 例 manifest；H1 ReSplat 能力备份到 N1 | `h1-prep`（task-1） | 进行中 |
| E2 | **步数 Pareto 扫描**：13 例 × S∈{500,1k,2k,3k,5k,8k}（78 格），交付配方其余不变、schedule 随 S 匹配；3 卡并行出质量，再单卡串行校 B 秒 | `n1-sweep`（task-2） | 待 E1 放行 |
| E3 | Pareto 分析 → 每例最优 S / 统一最优 S / 自适应停止规则（train-only 信号，边际阈值 ≈ `20/(0.9·budget)`≈0.22–0.24 分/秒）；修 `run_pipeline` realloc 默认 | `analyst`（task-3） | 分析器先行 |

### 为什么 E2 值得做（预期量级，不是承诺）
TIME 项满分为 20；当前 delivery 的 B=22~36s、预算 92.6~99.5s，对应 TIME≈13.6~17.3。易例若质量在 1000–2000 步即饱和，砍到该步数可把 TIME 推向 19~20（**每例 +2~5**）而质量几乎不掉；难例保持 5000 甚至加步仍是正收益。**13 例均值预期 +1.5~+4**，而且它是 E3 自适应策略的基础；这是当前「大颗粒」实验里唯一还没测过、且不需要新模块的一条。

---

## 5. 不变约束（违反即无效）

1. ReSplat 权重/网络冻结，`num_refine=2`，最多 16 帧、单次前馈。
2. 预算 = 同卡类随机初始化 FastGS **30000 步**实测秒数：dl3dv 94.29、hw/billboard 99.49、其余 hw 95.0、mip bicycle/bonsai 95.0、counter 92.57。
3. **TIME 项只计模块 B**（已与用户确认）；A/ReSplat 不计时。
4. 计时必须单卡串行；并发读数只能用于质量。
5. **评测分辨率 = 原始分辨率（用户 2026-10-30 明确确认；2026-10-31 进一步对齐实现口径）**；不得再尝试降分辨率。
   准确含义（已逐行核对官方代码）：**A 输出分辨率 = ReSplat 输入 = 模块 B 后优化分辨率 = 评测分辨率，四者一致**。
   - `module_a/preprocess.py:330-357`：每张图 LANCZOS 缩放到 `max_long = 768(竖屏)/960(横屏)`、H/W 为 64 倍数，内参逐轴缩放（`:364`），写入 `split.json["resolution"]`；
   - `run_resplat.py:217`：`target_h, target_w = split["resolution"]`，ReSplat 就在该分辨率前馈；
   - 模块 B：`Scene(-s <work>)` 直接读 A 的图（`train_from_ply_fastgs.py:87` / `research/train_sweep.py:219`），训练渲染与监督都用 `viewpoint_cam.original_image`（`train_sweep.py:385`）；交付配方不传 `--res_scales` → `scales=[1.0]`（`train_sweep.py:214`），即**后优化就在该分辨率做，没有二次降采样**；
   - 评测：`render_test.py` 渲染测试视角后与 `cam.original_image` 比较，`evaluate.py:53` 断言 `rendered.shape == gt.shape` —— 渲染与 GT 必须同分辨率。
   因此：**不需要**按数据集原生像素（例如 mipnerf360 的 4946×3286）重评；本仓库 13 例数字就是这套官方口径下的分数。
6. 不删赛方原始数据集；大文件一律落 `/root/gpufree-data`（N1 根盘 99%）。
7. 不把旧 GPU/旧 pod 的秒数拼成同机实测；跨 GPU 的分数只能叫估算。

---

## 6. 文件与证据索引

| 内容 | 位置 |
|---|---|
| 13 例配对结果（落盘） | H1 `outputs/scene13r15/results.jsonl` |
| 13 例执行脚本 | `research/experiments/scene_eval_r15.sh` |
| 3 难例步数曲线 | H1 `outputs/ocurve/*/s*/summary.json` |
| 旧会话研究叙事（**§1–§55**） | `docs/RESEARCH-20261030.md` |
| 13 例评测节点用法 | `docs/EVAL-NODE.md` |
| 旧节点盘点（已知过时项见 §1） | `../GPU_NODES_20261030.md` |
| 交付入口 | `run_pipeline.py`、`DELIVERY.md` |
| 本会话新增工具 | `../scripts/remote.sh`（H1/N1 重试 SSH） |
