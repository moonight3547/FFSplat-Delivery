# E5-G：固定拓扑后的按视角反传门控

调研：2026-10-30 06:50 +08:00。新学术备选，尚未实现或运行。它与E5视角重采样、E8-L降optimizer频率、E8-F执行后端不同，分别裁决。

## 一手进展与相关性

[SkipGS，ECCV2026，论文v2](https://arxiv.org/html/2603.08997v2)，[官方代码](https://github.com/ASU-ESIC-FAN-Lab/SkipGS)，[FastGS适配说明](https://github.com/ASU-ESIC-FAN-Lab/SkipGS/blob/main/examples/integrate_fastgs.md)。结构停止后始终做forward，用每视角loss相对历史EMA的比值决定是否backward，并用最低反传预算防止过量跳过。论文在FastGS/Mip-NeRF360上报告训练181.9→164.5秒、后结构87.2→69.8秒，质量指标略退；不是只对原始慢3DGS有效。

但实验是30K/15K后精修、RTX Ada5000；不是冻结ReSplat的5K短跑。论文所称end-to-end是其trainer范围，不能当成本项目A+R+B已提速。当前代码main还包含可选adaptive step/visibility union等扩展，首轮不一起启用；实现前由coding agent固定具体commit。

## 本项目先要证伪什么

假设：结构截止1250后，一部分重复训练视角的backward边际价值较低，保留forward观察和大部分更新即可节省时间。比只省optimizer，反传门控能触及更大的成本部分；但不能假设已有“收敛”。round-new→round-long的counter还增加3.41Q，说明后期更新仍有价值。

loss低于EMA不等于loss已经平台：持续单调下降的loss也会低于历史EMA，因而被提出跳过。需要看实际跳过率、loss下降和最终质量，不能把每个skip称为已证明无用的梯度。

## 两阶段最小实验

**G0，shadow诊断**：在当前C的正常训练中只记录门控本来会如何决策，不改变backward。使用原有训练loss，不渲染test，不额外渲染全数据集。记录每图访问次数、EMA、would-skip、强制反传原因、最后1250步的loss变化和真实backward成本占比；统计日志不要逐步强制CUDA同步。若几乎没有可跳过量，或节省上界太小，停止。

**G1，保守单候选**：前2000步完全同C；接着500步仅校准/观察，最后2500步才可门控。EMA decay=.95、阈值0、最低反传比例先固定.75；这些是本地保守适配值，其中.75和延迟启用不是论文默认。所有场景统一，不开阈值网格。C与G共用相同forward视角序列、5K总forward、LR计划、结构/opacity/SH设置，首轮不把省下时间换成更多步。

若支持原文auto预算作为后续机制对照，另开组，不悄悄替换上述固定.75。预算累计通常包含warmup，所以“总反传比例≥.75”不等于门控窗口单独≥.75；必须分别报告。首轮不启用adaptive_norm累积、E8-L、E3等同时改变更新的因素。

## FastGS集成需要明确的边界

作者通用示例的整段continue不应直接套到本train_sweep。它可能跳过快照、预算检查、最终导出，也会干扰rest-SH每16步的累计更新。作者专门的FastGS说明要求skip-only模式仍按原global iteration调用native optimizer_step，让已累计SH梯度在原定时钟处理；稀疏优化器分支另有规则。首轮应遵循本项目实际两个Adam的状态语义，而不是统一在skip时清空所有grad。

主geometry正常step后已zero_grad(None)，skip时应没有旧geometry梯度被误用；SH可能有过去步骤的待处理梯度，不能丢弃。门控不得在结构阶段生效，任何读取.grad的结构hook必须拿到该步有效梯度。LR与总forward时钟继续按预注册规则前进，不从测试分数决定何时恢复反传。

## 裁决

记录实际skip比例、反传/step次数、SH待处理与flush次数、峰值/最终N、原始PSNR/SSIM/VGG-LPIPS和交付墙钟。先C/G成对；只有有净收益才补匹配反传预算的随机门控，区分“按视角选择有用”与“少做一些更新就足够”。随机控制使用独立RNG，不改变相机序列。

若训练loss仍快速下降、质量回退抵消节时、或CPU门控/同步开销吃掉收益，就降优先级。E8-L失败不等于此路线失败；同样，此路线失败也不能证明后期所有梯度都不可省。暂列低成本备选，SH导出及真实计时核查优先。
