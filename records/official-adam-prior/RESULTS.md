# CVPR2026 Faster-GS官方optimizer组件：12模型完成

2026-10-31 05:05收口。N3三场景×两seed×foreach/官方CUDA Adam共12个fresh B全部完成，并通过独立CPU审核。**完整B逐配对平均节省7.04%，三原质量指标均值小退，宏B-only代理32.30350→32.55633，净+0.25283。** 保留为有速度收益的研究组件；不称无损加速，不切换默认，不把它当完整Faster-GS复现。

## 原始质量与完整B

|配方|均PSNR|均SSIM|均LPIPS|内部B秒|完整B秒|宏Q80|TIME20|宏B-only/100|逐caseclip总分均|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|原foreach C0|24.04667|.778767|.229367|36.9807|40.0526|17.66533|14.63817|32.30350|32.84684|
|官方CUDA Adam组件|23.96550|.777733|.230017|34.1420|37.2390|17.38773|15.16860|32.55633|33.10300|

质量宏代理先平均六组原始PSNR/SSIM/LPIPS再clip；TIME逐case/seed按完整B进程与同GPU同范围随机30K预算计算后平均。逐caseclip口径另列，不能混用。TIME只计完整B，是主范围；A/R/E2E仅作成本诊断。完整百分制和13例官方聚合仍未确认，official_score_100=null。

六配对平均ΔPSNR−.08117dB、ΔSSIM−.001033、ΔLPIPS+.000650；宏Q80−.27760，TIME+.53043，净宏代理+.25283，逐caseclip总均净+.25617。完整B均值少2.814秒；逐pair时间比例92.9617%，内部B比例92.3094%，并非论文整套框架的2–5倍。

|场景|seed|foreach B秒|官方Adam B秒|时间节省|ΔPSNR|ΔSSIM|ΔLPIPS|
|---|---:|---:|---:|---:|---:|---:|---:|
|bicycle|2|38.380|35.450|7.63%|+.019|+.0002|+.0005|
|bicycle|3|38.464|35.356|8.08%|−.015|.0000|−.0004|
|DL0b|2|39.656|37.722|4.88%|−.104|−.0006|+.0006|
|DL0b|3|40.579|36.753|9.43%|−.188|−.0022|+.0006|
|DL3e|2|41.494|39.322|5.24%|−.047|−.0009|+.0004|
|DL3e|3|41.743|38.831|6.98%|−.152|−.0027|+.0022|

完整B为6/6更快，LPIPS为5/6退化、1/6改善；不能由均值+.00065掩盖DL3e seed3的+.0022。两个seed不足以建立时延置信区间。最终高斯数均值726,126.8→725,700.8（−426，约−.059%）；原生结构日程相同，实际点集会因数值优化路径发生变化。

## 官方身份、实现与烟测

本项目为Florian Hahlbohm等[CVPR2026 Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting/)，官方main `d23bb784d8f19b2432d847b80418d56f0494f599`。它与此前GoogolplexGoodenough Workload-Balanced FasterGS为不同项目。本轮直接使用官方`FasterGSCudaBackend.torch_bindings.adam.FusedAdam`，不是旧PyTorch `fused=True`，也不是`FasterGSFused`分支的backward+optimizer融合。

训练源码`ade0eae51ee1b99ec540cff9741709742d686dbf`；远端`/root/faster-gs-cvpr2026/src/project`全程保持此HEAD。官方完整backend以`-O3 -use_fast_math -std=c++17`、SM86编译到隔离target `/root/faster-gs-cvpr2026/site`，Torch2.4.0/cu124；未改共享环境、protected upstream或旧同名backend工作树。wrapper只替换两个optimizer构造，六组参数、原FastGS renderer/score/AbsGS、SH16步累积、C0拓扑/损失/LR、末100步全分辨率VGG/.1不变。seed2 foreach先，seed3顺序反转；实际5000TRAIN序列逐项匹配同一冻结计划。

每卡先完成64步真实CUDA数值/状态检查：eps1e-15、β=.9/.999，非零/小梯度/零/None，append/prune后moment与step迁移；三卡最大参数绝对差均9.88133e−7。None不更新参数/时钟，零梯度继续更新时钟与moment。官方FMA/fastmath并非逐位相同，不能据此宣称质量等价。

每例两臂真实TRAIN600烟测均通过：600步native增密、VGG501–600共100次，signed/AbsGS四维screen gradient、六组叶参数/SH梯度有限；结构替换后的grad=None及原时钟符合预期。smoke只看TRAIN，没有评分TEST。

## 同GPU预算与独立审核

|N3 case/GPU|随机30K内部B秒|随机30K完整B秒|候选验收|
|---|---:|---:|---|
|bicycle/GPU0|125.169|128.2646|4/4两范围通过|
|DL0b/GPU1|109.152|112.1168|4/4两范围通过|
|DL3e/GPU2|110.504|113.5636|4/4两范围通过|

分母重新实测`4ba92ec`原PIL/optimizer/writer、无cache/wrapper、随机初始化FastGS30000；仅3000/6000/9000/12000 reset，29999次optimizer调度调用。不是ReSplat接续30K，不交叉使用N2或其它case秒数。

独立CPU审计源码`384f03dc96ff9249bddb329e7b55c848ac3b736d`，运行于隔离audit worktree，不改训练checkout。重新读取12正式PLY、6烟测PLY及3参考PLY，21个PLY全部属性finite；896张官方PNG完整RGB uint8解码并重算SHA。重新核验当前prepared/cache/初始R、N2冻结manifest与before/after完全一致；GPU UUID、5000TRAIN计划与SHA、实际native参数、六组Adam时钟、VGG权重SHA、官方class实际导入文件和.so SHA一致。候选每模型25,275次实际官方Adam kernel调用，几何五组state step4993、SH state step310；预算内外范围均已分开核对。

证据：[原始汇总及六配对](evidence/faster-gs-cvpr2026-20261031/raw/summary.json)、[12行CSV](evidence/faster-gs-cvpr2026-20261031/raw/summary.csv)、[独立审核](evidence/faster-gs-cvpr2026-20261031/independent-audit.json)、[官方构建日志](evidence/faster-gs-cvpr2026-20261031/build.log)。[预注册协议](FASTER_GS_CVPR2026_COMPONENT_20261031.md)、[renderer/locality可行性](FASTER_GS_CVPR2026_RENDERER_FEASIBILITY_20261031.md)保留独立范围。

## 使用边界与后续

本结果是三个badcase双seed，未证明全13净得分。如果仅这三例替换、其余十例不变，且全13宏质量分项处于未clip线性段，等权代理净贡献约+.05835；不是绝对总分预测。当前略正代理伴随质量下降，不据此全局切默认，且本轮没有与BigD2500叠加。

完整renderer还涉及投影、alpha截断、实际radii、AbsGS聚合统计及FastGS metric_map计数契约；官方Inria例修改了ADC，不能直接照搬假装相同接口。Morton/locality依赖per-Gaussian backward与全状态/稳定身份置换。下一步应先做明确的renderer兼容诊断，不能从本Adam组件的微小净收益或质量损失推断整套Faster-GS无效。

N3三卡在05:05实查均0%/1MiB；父队列72323/72324/72325、summary waiter与CPU审核已结束。完整PLY/PNG仍保留`/root/faster-gs-cvpr2026/outputs/component-20261031-v1`，没有删除唯一资产。小JSON、源码与审核证据已回传；GPU预留，后续只读库存盘点，不新开GPU网格。
