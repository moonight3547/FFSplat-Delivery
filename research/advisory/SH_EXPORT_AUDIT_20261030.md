# P0接口实验：世界系导出的SH旋转符号不一致

审查：2026-10-30 06:24 +08:00。只读固定源码，执行CPU代数检查；没有修改upstream、运行代码或PLY，没有运行GPU实验。

## 已确认的局部数学问题

`competition/upstream/baseline/run_resplat.py::eval_sh_basis` 一阶三项为 `[-c*y,-c*z,-c*x]`，c=0.4886025119029199；官方FastGS `utils/sh_utils.py::eval_sh` 和本地经典3DGS CUDA evaluator则是 `[-c*y,+c*z,-c*x]`。这里的z项符号不一致。独立一手代码亦见 [GraphDeco sh_utils](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/sh_utils.py)。

这个basis用于 `sh_rotation_matrix`，随后实际乘到导出harmonics。当前 `research.resplat_variant → competition.resplat.load_runner` 仅包装seed/build_model，所读代码没有覆盖该basis；因此它不是单独未使用的示例函数。运行时具体版本仍需对照远端文件hash。

[CPU原始审计](evidence/20261030-sh-rotation-cpu-audit.json)抽取固定源码纯数学函数，与官方FastGS evaluator比较1024个方向。仅设SH index2系数0.1时：

|全局旋转|当前导出的最大颜色误差|修正basis后误差|
|---|---:|---:|
|恒等|6.25e-17|6.25e-17|
|绕z轴90°|6.25e-17|6.25e-17|
|绕y轴90°|0.0977199|6.25e-17|

因此只做恒等/绕z旋转的检查会漏掉问题。原numpy运行在本机matmul/lstsq附近出现浮点警告，数值输出均有限；另保存了[不依赖numpy/BLAS的独立标量验证](evidence/20261030-sh-rotation-scalar-validation.json)：原方向+z、旋转后+x时，正确SH辐射为+0.04886025，当前旋转为−0.04886025，差0.09772050。加共同+0.5后RGB仍在[0,1]，这个反例不依赖颜色clamp。

**边界**：这证明局部导出旋转不能保持官方SH颜色不变；没有证明真实场景掉多少分，也没有证明5K后优化不能自行补偿。现有原结果仍是现有管线的有效观测，不重写为“全部错误”。

### 06:25远端实际版本与pivot核验

[H1只读核验](evidence/20261030-0625-h1-sh-pivot-audit.json)中，实际run_resplat.py与本CPU审计hash完全相同（`ecdcd3e3…498a`），并读取当前prepared split指定的同一pivot。真实pivot的一阶旋转算子差异Frobenius范数 `||D_bad-D_good||F`：DL3DV约0.2830、hw约0.4313、counter约2.5397。不是只有假设中的90°旋转才会命中；counter的算子差异较大。

这些数值仍不包含实际SH系数能量，不能按范数给场景预测分数，也不构成逐场景选择修正的理由。正确的basis转换应对全部场景统一应用；质量是否有净收益由F1/F2决定。

## 修正的数学边界，供coding agent处理

令S为16维单位阵，仅index2对角元为−1；所读basis满足 `Y_bad=S Y_official`。由坏basis构造的旋转 `D_bad`，对应官方basis的正确旋转为 `D_good=S D_bad S`。不能直接把最终PLY的index2统一取反，那不是相同变换。

若从已导出的错误系数恢复，应使用 `D_good D_bad^{-1}` 作用于旧导出系数，旋转R必须来自该次split的同一个context pivot，不能猜。更简单的路线是在本项目wrapper中仅替换导出basis后重新导出同一次推理张量。无论采用哪条，保持受保护upstream原样；新输出另存目录，保留旧hash。

不改变ReSplat网络、权重、refine数或context集合，不再调用额外ReSplat来择优。PLY与NPZ的SH必须同步修正，防止B/播放器读到不同资产；位置/scale/quaternion/opacity及高斯数量应一致。

本轮也核对了几何旋转路径：encoder及forward_update的covariance用context局部旋转构建后转到batch/world，固定导出再用原context c2w与local quaternion构造最终world quaternion；两端xyzw约定一致。当前未找到需要“再旋转一次几何”的证据；SH修正不要顺带对means/quaternion重复施加pivot。

## 最小实验：先接口，再效果

1. **F0**：单次已知旋转/系数检查，至少包括绕y或x的旋转；以实际FastGS/gsplat渲染颜色或其一致的basis evaluator验证。identity通过不足以验收。
2. **F1**：在已有三例初始化上，比较原导出和修正导出在固定少量训练视角的RGB一致性与原始训练误差；记录pivot R、SH一阶能量、每高斯颜色变化分位数。不要用test挑修正符号或pivot方向。
3. **F2**：只有F1确认实际命中后，用C_raw完整5K配方做成对比较；所有其他属性、seed、预算、GPU、计时一致，补正/导出耗时计入。接口修复若只改善0步但5K最终等价，结论为优化已补偿，而不是承诺额外分数。

这是比继续扩大step频率网格更靠前的初始化接口核查。它与raw/sharp不透明度A/B无关，首轮不要同时改变opacity或结构策略。

注意ReSplat decoder使用gsplat antialiased模式和eps2d=0.1，FastGS的实际渲染设置不同；即便修正SH，原生ReSplat图与官方PLY图也不应被要求无条件逐像素完全一致。F0先在相同SH evaluator上验证颜色旋转，F1再区分光栅化差异与坐标/SH差异，避免错误归因。

## 最新学术背景为何相关

[Frame-Consistent Feed-Forward 3D Gaussian Splatting，2026-10-16预印本](https://arxiv.org/html/2610.23039v1)在冻结VGGT/TokenGS的另一任务里指出坐标/尺度接口可主导表观质量损失。它提示先审查接口，但其×7尺度因子不适用于本项目，不能复制；其稀疏协议还使用target图像估计pose，信息范围与本项目要逐项区别。本文的SH问题来自本地源码和独立数学反例，不是套用该论文的收益。
