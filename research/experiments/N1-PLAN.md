# N1 下一阶段优化计划（底层加速方向）

H1 正在单卡串行跑 13 例新配方基准；N1 三卡用于**底层执行效率**探索——目标是降低模块 B 的墙钟，
在 TIME 占 20 分的口径下每省 5 秒 ≈ +1 分。

## 文献参考（已检索）

| 工作 | 机制 | 对本项目的可迁移性 |
|---|---|---|
| **Faster-GS**（CVPR 2026，[分析](https://raw.githubusercontent.com/zhaoyang97/Paper-Notes-en/refs/heads/main/docs/CVPR2026/3d_vision/faster-gs_analyzing_and_improving_gaussian_splatting_optimization.md)） | 系统分析 3DGS 优化瓶颈（fused Adam、SH 更新、致密化统计） | **最高**：给出的优化器/SH 改法可直接在 wrapper 层试 |
| **Speedy-Splat**（CVPR 2025，[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Hanson_Speedy-Splat_Fast_3D_Gaussian_Splatting_with_Sparse_Pixels_and_Sparse_CVPR_2025_paper.html)） | 稀疏像素 + 稀疏基元，渲染 1.9~4.7× | 中：需要改 CUDA rasterizer（受保护 upstream），但"先剪掉低贡献基元"可在训练前近似 |
| **FlashGS**（2025，[论文](https://ieeexplore.ieee.org/document/11093809)） | 大规模高分辨率渲染优化 | 低：我们分辨率固定 960×540，规模不大 |
| **3DGS-LM / 二阶优化**（[MPI](https://vcai.mpi-inf.mpg.de/projects/LM-IS/)） | 用 LM + 残差采样替代 Adam，收敛更快 | 中：短预算下可能有效，但实现成本高 |

## 实验队列（N1 三卡并行，优先级从高到低）

### P1｜fused Adam（最便宜，先做）
`torch.optim.Adam(..., fused=True)` 替换 FastGS `training_setup` 里创建的 Adam，其余不变。
- 预期：优化器占 B 的 ~22%，fused 通常快 10~25% → 总分 +0.5~1
- 判定：同场景同 seed 配对，B 秒数下降 ≥2 秒且质量不掉

### P2｜SH 通道裁剪（第二便宜）
训练期用 `sh_degree=2`（或更快 SH 升级日程），只在最后 500 步升到 3。
- 机理：rest SH 占参数量的 ~80%，其梯度是 backward 的主要成本
- 判定：质量掉 ≤0.3 质量分且 B 秒数降 ≥3 秒则采用

### P3｜可见性稀疏更新（真正的 sparse Adam）
只对当前视角可见的高斯做 Adam 更新（radii>0 掩码），维护完整 optimizer state。
- 与之前失败的"隔步 Adam"不同：不改更新频率，只改更新集合
- 需处理 SH 的 16 步累积窗口（取窗口内可见性并集）
- 判定：B 降 ≥3 秒且质量不掉

### P4｜bf16 混合精度（激进，最后做）
rasterizer 前后向用 bf16、参数保持 fp32。
- 风险：质量损失未知；只在 P1~P3 都做完后试
- 判定：质量掉 ≤0.5 且 B 降 ≥5 秒

## 执行约束
- 每个实验必须与「当前交付配方」配对（同场景、同 seed、相邻时段），避免 GPU 时钟漂移
- 计时必须单卡串行；三卡可以各跑一个不同实验，但不能在同一张卡上叠加
- 任何改动不改变：更新频率、学习率日程、损失权重、结构操作时点
