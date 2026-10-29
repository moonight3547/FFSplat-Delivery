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
4. 修复版depth-weight=0与原控制完全配对，然后才进入 EXEPERIMENTS 的全图/双mask对照。pose保持关闭。

这几项针对真实数学/接口风险，值得先验证；本轮不要求新增一套泛化测试框架，也不对尚未运行的深度方案预判增益。

## 5. 学术背景与限制

[In Depth We Trust](https://arxiv.org/html/2604.05715v1)强调选择性监督和不可靠先验的传播控制。本项目首先要让深度监督的值与梯度自洽；先验掩码无法挽救错误的归一化梯度。通过这些检查后，E3仍是“可靠先验 × 待修复区域”的弱监督方案，而非把所有像素锁到预测深度。
