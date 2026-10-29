# E8：结构稳定后，按可见性更新优化器

调研：2026-10-30 06:16 +08:00。研究方案，尚未实施。先读 [同步03](SYNC_20261030_03.md)：当前三seed证据支持保留前期结构操作，不能为了省更新而同时取消它。

## 学术依据与迁移判断

[A Step to Decouple Optimization in 3DGS，ICLR2026](https://eliottdjay.github.io/adamwgs/)，[论文v5](https://arxiv.org/html/2601.16736v5)，[实现](https://github.com/EliottDJay/3DGS_AdamWGS)把Sparse Adam、状态重缩放RSR、属性正则解耦DAR分开研究：dense Adam会让当前不可见高斯沿历史动量继续移动；稀疏更新省成本，也会改变探索。Sparse本身不保证同质量，DAR也不等于给所有参数套torch AdamW。

论文对照使用原生PyTorch Adam，实验规模、场景与本项目不同；附录C还显示室内场景更新成本较小。因而本项目先测optimizer占比，再决定是否移植。作者40%左右总时间改善不能直接套给FastGS。

首轮不引入DAR：当前控制没有相同的显式opacity/scale正则，直接搬它会同时改变目标函数。也不默认添加RSR，先分离最容易验证的可见性更新。

## 本项目的具体差别

已读固定快照 `competition/upstream/baseline/fastgs/scene/gaussian_model.py` 的 `training_setup` 与 `optimizer_step`：当前5K控制对xyz/DC/opacity/scale/rotation每步做dense Adam；rest-SH另有优化器，每16步更新并清梯度。这个16步累积日程不是按高斯可见性做sparse。

上游虽有 `optimizer_type=sparse_adam` 分支，却没有同时建立 `shoptimizer`；现有step路径又未传可见mask，不能仅换flag就认为支持该方法。应在本项目研究wrapper/子类适配，不修改受保护upstream。作者实现也明确向稀疏step传可见性参数，见 [train.py](https://github.com/EliottDJay/3DGS_AdamWGS/blob/main/train.py)。

## 最小实验

1. **诊断**：在C_raw已有训练中只统计固定少量窗口：forward/backward/optimizer/结构耗时；可见高斯比例；当前不可见但dense Adam实际位移非零的比例及位移分位数。CUDA计时不逐步强制同步，避免日志本身主导耗时。没有这份证据，不先写整套优化器。
2. **C**：C_raw统一5K、结构500–1250，保持已有SH更新日程。
3. **S**：前1250步完全同C；结构停止后的下一步才切可见性稀疏更新。保留参数、LR、rest-SH累积规则及最终导出。切换时保留已有m/v；不能悄悄reset状态。

必须预先定义“可见”：radii>0是覆盖候选，不等于有效像素贡献，也不等于非零梯度。首版可用已有radii mask，但日志明确其定义；不要为此新增全视角渲染。rest-SH跨16步应记录这段窗口的可见性并集，避免只以第16步mask扔掉先前视角的累计梯度。

逐高斯时钟与全局bias correction会产生不同更新语义，选用哪一种必须固定并写入结果。用一小组可见/不可见交替的参数序列核查状态继承即可；不需要泛化的大测试框架。确认切换点确实无后续split/prune，否则需同步维护参数与状态索引。

先做同seed、同输入、同GPU的C/S配对；只有质量不退且有可重复实际墙钟收益，才补一个新seed及RSR机制对照。E6几何限幅、E8稀疏更新、E3深度loss分别验证，首轮不组合。

## 裁决与止损

以原始PSNR/SSIM/VGG-LPIPS与实际交付墙钟为主，不从测试曲线选停止点。如果optimizer只占B的很小部分，按Amdahl上界不足以覆盖适配和mask开销，就停止该路线；即使更新kernel更快，也可能没有E2E收益。C/S总预算严格一致，不能因为少做更新而追加未记账的步数。

若S只变快但明显掉质，先检查状态时钟/SH并集，再考虑dense前段与稀疏后段的探索差异；不能把RSR当作必然修复。如果不可见更新本来很小、可见比例接近1、或mask/索引成本吃掉收益，直接降优先级。

## 06:20：新增诊断入口的口径建议

coding agent已提交 `dd266a7`，新增time_breakdown入口，尚非完成实验。当前标为optimizer的时间段从backward后开始，包含loss.item、致密化统计、结构操作及其额外多视角render，再到optimizer_step；不能把这一整段占比当成Sparse Adam可节省占比。`grad_to_step`实际主要是GT加载与loss forward，也应按语义阅读。

建议至少分开structure/statistics与真正step，并单独报告1250步后固定拓扑窗口。当前每步四处CUDA同步会改变流水重叠，profiling训练秒数不直接用作交付出分；先获得机制占比，再以关闭profiling的正常运行核对实际净收益。无需为了占比重复做质量评测。

## 与真正appearance-only的关系

E8仍更新可见高斯的几何，不是只拟外观。真正appearance-only应显式冻结xyz、scale、rotation，说明opacity是否仍训练，且保留前期结构阶段；即便冻结参数，渲染器backward仍可能计算几何导数，不承诺速度收益。历史 `--no_densify` 没有冻结几何，不构成该实验的负证据。
