# FFSplat-Delivery baseline

顶层结构对应官方 `cb29814`：module_a / run_resplat.py / module_b / fastgs / 3dgs / 训练、评测与参考脚本；原始完整版本保存在 ../competition/upstream（不修改）。FastGS 和 ReSplat 源码复制时保持逐字节一致，未修改网络/原生算法。最终 B 的真实实现是 ../research/train_confirmed.py。

```bash
# 默认官方 scenes.json 三组；先预览，不训练
bash run_train.sh --describe
# 只跑 bicycle 或任一单场景；output 需空目录
bash run_train.sh --scene /data/mipnerf360/bicycle --output results/bicycle
# 已有单场景输出的正式重新渲染、评测
bash run_eval.sh --output results/bicycle
```

默认 conda 环境 resplat / fastgs，PATH 中提供 conda；不同机器使用 --profile 覆盖。权重放 ../resplat/pretrained，数据放 ../datasets/examples。run_train.sh 包含完整 A/R/B 和正式评测；产物在每个 output 内的 work/、results/{resplat,module_b}/、report.json。run_reference.sh 接收 -s <prepared> -m <model>，默认 native 随机 FastGS30K，区别于官方原文的经典3DGS参考；计时按用户确认的完整B范围。

---

以下为官方 baseline README 历史原文，算法成绩/默认10K/经典3DGS参考不代表本提交默认。

# ReSplat 竞赛 Baseline（A → ReSplat → B）

经典三段式基线：**模块 A**（位姿获取 + 均匀 16 视角选择）→ **ReSplat**（16 视角前馈重建）→
**模块 B**（FastGS 接续训练 10000 步）。每阶段独立计时，最终输出
PSNR / SSIM / LPIPS + 总耗时。`datasets/examples` 三组数据各跑一个场景，一键运行。

## 一键运行

```bash
# 0) 一次性环境准备（编译 baseline/3dgs 的 CUDA 子模块，需要 fastgs conda 环境）
bash setup.sh

# 1) 一键端到端训练：A -> ReSplat -> B（三组数据各一个场景）
bash run_train.sh              # 全部；或指定组：bash run_train.sh dl3dv hw

# 2) 一键评测：渲染测试视角 + PSNR/SSIM/LPIPS + 时间汇总表
bash run_eval.sh               # 同上可指定组

# 可选）经典 3DGS 30k 从头训练参考（选手需要打败的时间线）
bash run_reference.sh
```

产物：
```
work/<group>/<scene>/            # 模块 A 产物（COLMAP 格式场景 + split.json + module_a.json）
results/<group>/<scene>/
  resplat/                       # resplat_world.ply/.npz + resplat.json（计时）
  module_b/                      # 训练模型 + module_b.json（计时）+ metrics.json
  reference_3dgs/                # （可选）30k 参考
results/summary.{json,md}        # 汇总表
```

## 环境

- **resplat 环境**（模块 A / ReSplat / 评测指标）：按 `../resplat/README.md` 安装
  （python 3.12, torch 2.7, gsplat 1.5.3, pointops），并安装官方权重：
  `resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`
  （选手从主办方提供的链接下载，放入 `../resplat/pretrained/`，或用
  `run_resplat.py --checkpoint` 指定路径）
- **fastgs 环境**（模块 B / 渲染）：torch 2.4 + pip 安装的
  `diff_gaussian_rasterization_fastgs` / `simple_knn` / `fused_ssim`，
  并 `bash setup.sh` 编译 `baseline/3dgs/submodules`（供 30k 参考计时用）
- **数据集**：选手从主办方链接下载后放入 `../datasets/examples/<group>/<scene>/`
  （`scenes.json` 配置每组使用的场景）

## 已验证的基线结果（RTX 4090 24GB，从零一键重跑）

| group | scene | PSNR | SSIM | LPIPS | A(s) | resplat(s) | B(s) | **总耗时** | 经典30k参考(s) | 参考PSNR |
|---|---|---|---|---|---|---|---|---|---|---|
| dl3dv | 001dccbc…5f | 32.57 | 0.950 | 0.153 | 31.5 | 18.0 | 39.5 | **89.0** | 546.7 | 29.91 |
| hw | billboard | 22.68 | 0.799 | 0.159 | 89.6 | 15.4 | 37.7 | **142.7** | 915.1 | 16.75 |
| mipnerf360 | counter | 27.77 | 0.888 | 0.207 | 54.6 | 20.2 | 41.0 | **115.8** | 531.0 | 27.08 |

- 三组数据上，基线管线比经典 3DGS 30k 从头训练（随机初始化、同一 A 产物）
  **快 4.6~6.4 倍且 PSNR 全部更高**——基线已满足比赛目标；
  选手需在此基础上继续压缩时间 / 提升质量
- 参考 `benchmarks/gs_compare_20261020`：billboard 上 resplat16 单模型 18.25、
  FastGS（三角化点初始化）19.13；本基线（resplat 初始化 + FastGS 接续）为 22.68
  （评测视角划分不同，仅供参考）

## 流水线设计（baseline 的具体选择）

### 模块 A（`module_a/preprocess.py`，resplat 环境）

1. **位姿获取（有则跳过，无则估计）**，按来源优先级：
   - `sparse/0/{cameras,images}.{bin,txt}`（mipnerf360）→ 直接使用
   - `transforms.json`（DL3DV）→ 转 COLMAP（OpenGL→OpenCV 轴翻转 + 内参缩放，不估计）
   - `frames_info.txt`（hw 采集数据）→ **重新跑 COLMAP**（该文件的位置字段经独立
     COLMAP 重建交叉验证为几何无效：四元数为 W2C 且误差仅 0.54°，但位置与任何
     刚体对齐都不符）
   - 什么都没有 → 跑 COLMAP
2. **COLMAP 配方**（hw 路径）：`OPENCV` 相机模型估计畸变 + `image_undistorter`
   去畸变（hw 实拍存在 k1≈0.043 的真实畸变，直接按 PINHOLE 拟合会把畸变吸进
   位姿造成漂移——实测影响 ReSplat 自渲染 ~5 dB）；≤120 张用穷举匹配保证全局一致性
3. **分辨率**：长边 ≤ 960（横屏）/ ≤ 768（竖屏）且 H/W 为 64 的倍数
   （LANCZOS 缩放，内参逐轴缩放），全流程统一使用该分辨率
4. **竖屏处理**：图像保持原方向不旋转；`resplat/src/model/encoder/unimatch/
   mv_unimatch.py` 的 `assert ori_h <= ori_w` 按 benchmark 同款方式对竖屏放宽为
   no-op（对横屏行为不变）
5. **划分**：llffhold=8（与经典 3DGS 一致，每 8 张留 1 张作测试）；
   16 个 context 视角从**训练集**中 `np.linspace` 均匀选出（不接触测试视角）

### ReSplat 阶段（`run_resplat.py`，resplat 环境）

- `dl3dv_16v_540x960` 官方权重；context = A 选出的 16 张，
  refinement target = 同样 16 张（自监督精修，不向 ReSplat 输入额外视角）
- `infer_colmap.build_batch` 会把位姿归一化到中间 context 视角（pivot），
  高斯位于该归一化坐标系；导出时需要**完整逆变换**（本 baseline 的关键修正，
  已用渲染 PSNR 平价验证）：
  1. means：`X_world = pivot_c2w · X_norm`
  2. rotations：ReSplat 的 decoder 用**预计算协方差**渲染——
     `cov = R_ctx_norm[v] · R_local · Σ · R_localᵀ · R_ctx_norm[v]ᵀ`，
     即每个高斯的世界旋转 = **源 context 视角的世界旋转 × 局部旋转**
     （高斯按 v-major 排序，第 g 个的源视角 = g // (N/16)）
  3. SH 系数：光栅器在世界系求视线方向，刚体旋转后系数需做
     **Wigner-D 旋转**（按 pivot 旋转，逐阶块对角）
  4. far 平面剔除 + opacity<0.005 剪枝（对齐 ReSplat 渲染行为与
     模块 B 自身的剪枝阈值）
- 导出**经典 3DGS PLY 布局**（f_rest 按 channel-major 展平，与
  save_ply/load_ply 往返一致），同时保存 npz 便于调试

### 模块 B（`module_b/train_from_ply_fastgs.py`，fastgs 环境）

- **FastGS**（多视角一致致密化/剪枝 + tile 裁剪渲染器），复刻
  `benchmarks/gs_compare_20261020` 的配方：标准 30k LR 日程原样保留
  （position_lr_max_steps=30000），**10000 步早停**；FastGS 默认超参
  （mult=0.5、densification_interval=100、grad_abs_thresh=0.0012 等）
- **全训练视角**（llffhold 划分的 train 集）接续训练，初始化 =
  ReSplat 世界系 PLY（`--init_ply`）；`--init_random` 则从随机视锥点云开始
  （供经典 30k 参考计时用，`module_b/train_from_ply.py`）
- 10k 步仅 ~40s（~235 it/s，RTX 4090）

### 评测（`render_test.py` + `evaluate.py`）

- fastgs 环境用 **FastGS 原生渲染器 mult=1.0**（全 tile，benchmark 同款）
  渲染测试视角 → resplat 环境**官方 ReSplat 指标实现**
  （PSNR/SSIM/VGG-LPIPS，与榜单同源）对比 GT → `metrics.json`
- `summarize.py` 汇总各阶段计时与指标

## 关键调研结论（写代码前的验证工作）

1. **hw `frames_info.txt` 的位置字段不可用**（见上）；实际 jpg 为 720×960
   （文件声明的 1080×1440 是传感器尺寸），逐轴内参缩放自动兼容
2. **位姿写回的旋转-四元数转换必须 roundtrip 校验**：早期版本 rotmat→quat
   的解包顺序把每个相机旋转写成了逆旋转，导致所有场景 PSNR 掉 10~13 dB
   （现用 scipy 实现并加了 1000 组随机旋转的 roundtrip 测试，误差 <1e-15）
3. **ReSplat 高斯的世界系导出**是本 baseline 的核心难点：除 pivot 反归一化外，
   还有源视角旋转合成（decoder 用预计算协方差渲染，`rotations` 字段是局部旋转）
   和 SH 系数的 Wigner-D 旋转（光栅器在世界系求视线方向）——修复后世界系 PLY
   与 ReSplat 自渲染达到 PSNR 平价
4. 3DGS 经典代码在 sparse 只有 cameras/images（无 points3D）时崩溃——
   vendored 副本已修补（本管线的初始化本就不需要 SfM 点云）

## 选手可改进的方向（baseline 刻意保守之处）

- A：更快的位姿（免 COLMAP：利用 frames_info 的有效四元数 + 三角化恢复位置；
  或学习式位姿）、更聪明的 16 视角选择（FPS/覆盖率/信息量）
- ReSplat：更多 refinement 轮次、把训练视角作为 refinement target、分辨率/分块策略
- B：更少步数（如 3k-5k + 更高学习率）、蒸馏 ReSplat 初始化、致密化预算削减
- A+B 融合：增量式（ReSplat 出粗模型 → B 边训边补视角）
