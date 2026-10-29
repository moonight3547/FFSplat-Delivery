# 当前题意与执行约定

整理日期：2026-10-29。目标：在固定 ReSplat 前后添加模块 A / B，在时间限制内提升3D高斯重建质量。新对话以本文为工作入口；有冲突时保留来源并明确标注，不将推测改写为赛方原话。

## 1. 来源与优先级

1. **用户最新确认**：时间预算以随机初始化 FastGS 在同机训练30,000步的实测耗时为准。不是经典3DGS，不是从ReSplat输出接续30K；也不是10K耗时乘三或前20步外推。
2. **赛方当前代码/说明**：`https://gitcode.com/gcw_yxxmv6uE/hackathon_competition`，快照 `cb29814af7e260f03f7565e6edadb35bf42c5a93`。相对此前 `a1139a7` 仅更新 `baseline/README.md`，可执行代码无差异。原文见 [总README](competition/upstream/README.md)及[基线说明](competition/upstream/baseline/README.md)。
3. **原始题面**：[original-problem.txt](references/original-problem.txt)，原样保存，包含较早的经典3DGS/B-only约束及评分区间，不当作覆盖最新约定的文件。

**尚存冲突**：赛方当前README和reference脚本仍以经典3DGS30K为参照；用户要求FastGS随机30K。本仓库按用户要求执行，并在报告里保留 `reference_trainer=fastgs`。不得宣称该变更已获出题人确认。

## 2. 固定管线与可改范围

多视角图像/相机输入 → **A**（位姿、预处理、16视角选择）→ **ReSplat**（固定模型）→ **B**（高斯后优化）→ PLY、PNG和报告。

- 不训练或替换ReSplat，不改变网络结构、权重；当前调用采用赛方 `num_refine=2` 和既有竖屏补丁。
- A/B可以改策略，但超参数跨场景统一，不能人工逐场景调优；测试集RGB、LPIPS等不进入训练、选图或停止逻辑。
- 默认16个context来自训练集；`llffhold=8`。赛方A在划分前可能对全部输入图片做COLMAP，必须如实记录这一位姿阶段行为。
- 图像沿用赛方统一预处理：横屏长边≤960、竖屏≤768，尺寸为64倍数，内参随之缩放；**不是旧实验256×448**。参考和候选必须用相同处理结果。
- 模块B的训练图为全部训练视角，最终使用赛方FastGS完整tile渲染及官方VGG-LPIPS评估。

## 3. 默认基线与实验数据

O0 = 均匀16图 + 固定ReSplat + FastGS接续10,000步，保持30K位置LR调度。新参考 = **随机视锥高斯（默认目标约100K，实际记录初始化数量） + FastGS30,000步**，使用相同场景、train/test划分、处理分辨率与GPU。

默认三个case：DL3DV `001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f`、hw/billboard、mipnerf360/counter。完整赛方下载包含13场景；可选配置在 `benchmarks/`，不与默认三例平均混用。

## 4. 时间与评分

保存A、ReSplat、B、E2E以及各阶段进程墙钟；脚本内部计时和进程墙钟各自与对应参考比较，不交叉混算。参考测量本身不加入候选时间。当前主报告沿用赛方更新后的 **A+ReSplat+B** 范围，同时展示 **B-only** 诊断。分子范围尚不能从“FastGS30K”这一分母更正中自动推导为改变。

`T_budget = 实测 FastGS随机初始化30,000步 total_seconds`。所选范围时间 `T >= T_budget` 即无效。环境、GPU UUID、依赖、输入、代码必须可追溯；不能把不同GPU的历史秒数拼成同机实测。

原题的线性百分制作为研究代理分（clip表示限制在区间）：

- PSNR：`20 × clip((PSNR-15)/25, 0, 1)`。
- SSIM：`20 × clip((SSIM-0.70)/0.25, 0, 1)`。
- LPIPS：`40 × clip((0.25-LPIPS)/0.20, 0, 1)`。
- TIME：有效时 `20 × min(1, (1-T/T_budget)/0.9)`；≤10%预算为20分；达到预算即无效，不能只给0分后继续报有效总分。
- 质量先取场景均值再计算分数；TIME逐case算分后平均。任何case无效，完整集合不能报有效总分。跨case官方聚合方式尚未确认。

新codebase没有明确重新确认完整百分制及TIME计分分子；`official_score_100=null`。报告必须标明“代理分 / B-only或E2E / 实测或估算”，并同时展示原始指标与分项得分。

## 5. 当前策略与优先级

- 已迁移A：ranked16。历史三例均没有改变选中context，尚未证明有效。
- 已迁移B：MCMC + coarse-to-fine（0.5/0.75/1）+ 随可用时间推进的LR、拓扑调度。历史质量有提升，但以错误的经典30K预算运行，不能直接宣称新口径得分提升。
- 首先完成正确FastGS预算与同GPU基线，再考虑缩短B、快速FastGS接续、降低A/COLMAP开销。不要在无效预算上继续堆训练轮数。

## 6. 交付与环境

输出可加载的PLY、测试PNG、质量/计时/输入和代码身份报告；最终需包装原题单场景一键入口。Conda管理环境；Mac M4和Windows作为控制端，实际CUDA测量在Linux 3090及以上GPU。新硬件通过profile接入，必须重新标定；不把Mac渲染速度映射当作真实GPU验收结果。
