# E3 前置审查：先修正深度的梯度与语义

2026-10-30 06:07 +08:00。只读代码与标量代数检查；没有运行GPU、没有实现修复。

## 1. camera-Z 与坐标单位：本地源码支持的部分

ReSplat `src/geometry/projection.py::get_world_rays` 把相机射线除以其z分量，而不是单位长度归一化；encoder随后 `mean=origin+direction*depth`。这支持其depth是相机Z的参数化，不应再当成欧氏射线距离做额外乘除。

`infer_colmap.py::build_batch` 的pivot归一化是刚体变换，本段没有场景尺度归一化；相机Z在世界刚体变换后仍对应相机局部Z。另行出现的data_shim/裁剪、预测分辨率与输出字段，仍需真实导出时检查。

`visualization_dump['depth_fullres']` 在初始encoder前馈时从 `depth_preds[-1]` 构造，必要时插值至输入大小。它不是后面两轮Gaussian refinement的最终渲染深度。可以作为同源预测先验，但不能称“refine2后的独立几何真值”。若归档时要说明先验stage，应写清 encoder initial depth head。

## 2. 已发现的梯度反例

`research/depth_reg.py::decode_depth_rgb` 当前执行：

`depth = numerator / alpha.clamp_min(eps).detach() * encode_max`。

其注释称detach分母可阻止opacity取巧，但对最简单单射线/单高斯情况，令 `N=a*z`、`A=a`，正确归一化深度 `D=N/A=z`，对a的导数应为0；只detach分母后，自动微分会给出 `dD/da=z/a`，制造并不存在的深度变化方向。

本轮CPU标量例：a=0.4、z=3，真实有限差分导数0，detach分母的代理导数7.5。原始检查：[JSON](evidence/20261030-depth-gradient-scalar-check.json)。这是代数反例，不是GPU端到端实验，也不说明已完成旧实验的低分一定由它导致。

即使loss再用detached alpha作为confidence权重，也不会恢复这个应为0的导数。几何移动、尺度改变所引起的footprint变化也可能受到同类错误归一化梯度影响。

## 3. 给coding agent的修正边界

首先明确监督量是 expected normalized depth `sum(wz)/sum(w)`，还是 unnormalized accumulated depth `sum(wz)`，并确保native depth与编码depth两条分支一致。当前native分支直接返回depth，而编码分支除alpha，接口注释又都叫alpha-weighted，有混用风险。

若要优化真正的expected depth，应保留分子/分母的一致可微关系；alpha有效性mask可以作为非可微筛选。若希望深度项完全不更新opacity，应该显式控制**深度项到opacity参数的梯度路径**，而不是仅detach归一化分母。保留光度项正常更新opacity。

不要悄悄更改depth的值定义，同时继续引用旧loss权重和旧结果；需要一个修复后的零深度权重control，确认训练/渲染行为和原FastGS一致。

## 4. 最小验收与新的E3执行顺序

1. 单高斯、单射线的normalized-depth值与alpha梯度检查；双层遮挡例核对expected/accumulated定义。
2. 固定已知相机和少量高斯，比较编码RGB解码的alpha、camera-Z和值域，检查实际CUDA中的颜色clamp与SH约定。
3. 真实sidecar做图像名、尺度、分辨率、camera-Z语义检查；同时保存先验stage、字段名及生产代码hash。
4. 修复版depth-weight=0与原控制完全配对，然后才进入 EXPERIMENTS 的全图/双mask对照。pose保持关闭。

这几项针对真实数学/接口风险，值得先验证；本轮不要求新增一套泛化测试框架，也不对尚未运行的深度方案预判增益。

## 6. 06:25补充：alpha不是先验正确概率，双mask各自负责什么

现有loss把渲染alpha命名confidence；它衡量当前GS的覆盖/透射，不衡量ReSplat depth预测是否正确。高alpha也可能是错误前景遮挡，低alpha可能是尚未覆盖的真实区域。只乘alpha不会自动实现“可靠先验 × 待修复区域”。

冻结ReSplat encoder内部确有 `results_dict['match_probs'][-1]` 沿深度维取max后作为head输入；这是多视图匹配分布峰值，不是经本场景校准的正确概率，也不是最终两轮refine后的depth置信度。当前depth sidecar未保存它。如需要，只在同一次推理用旁路hook读取深度预测器输出，不增加推理/改变网络；先检查真实输出字段、尺度和图像顺序。

**先验可靠mask**：优先用16个context之间现有camera-Z预测做少量邻接重投影与相互检查；只比较正深度、投影在界内且非明显遮挡的区域。遮挡/出界/缺观测应标为unknown，不自动当错误；repeated texture和共同尺度偏差也可能相互一致，所以此mask仍是启发式。match-prob可作独立排序信号，首轮不叠加多个未经证实的阈值。

**模型待修复mask**：以当前GS的几何不一致/稳定训练残差定位需要干预处；与先验相差大只说明分歧，不能单方面断言GS错。In Depth We Trust的DIM通过额外虚拟视角深度渲染检测模型不一致，不能把“预测和渲染之差”改名为DIM复现。每步双深度渲染成本也不适合直接搬进本项目5K短跑；首轮考虑固定少量检查时刻，或复用已有邻接训练视角的缓存深度。

**支持量门槛**：目前sidecar只覆盖16个context，B则用全部训练视角。已有记录hw为56个训练视角、counter为210，均匀采样时可命中depth的比例上限约28.6%与7.6%。相同lambda不等于相同全训练过程监督强度。实际有效像素比例还会进一步降低。日志应记录有sidecar步数、有效mask像素和累计有效权重；不盲目乘逆命中率放大loss，也不能把几乎从未命中的loss判作depth先验无效。

初步顺序仍为：修正值/梯度接口→零权重控制→记录上述支持量→全图与可靠×修复mask对照。若有效交集接近空，保留“先验覆盖不足”的结论，先完善诊断，不无限放松阈值。

### 07:44新增：已有SfM观测是另一种先验来源

[E3-S](SPARSE_GEOMETRY_ANCHORS_20261030.md)已从原始images.bin训练观测完成候选点重建及留一训练观测检查，无需ReSplat同源depth或新深度模型。它可在非context训练图提供稀疏camera-Z锚点，但不是外部GT。先检查现有0步/5K是否真偏离；命中后才做弱监督，仍须先修复归一化梯度、通过零权重控制。低重投影误差不等于全场景正确，稀疏loss也不等于稀疏渲染加速。

## 5. 学术背景与限制

[In Depth We Trust](https://arxiv.org/html/2604.05715v1)强调选择性监督和不可靠先验的传播控制。本项目首先要让深度监督的值与梯度自洽；先验掩码无法挽救错误的归一化梯度。通过这些检查后，E3仍是“可靠先验 × 待修复区域”的弱监督方案，而非把所有像素锁到预测深度。
