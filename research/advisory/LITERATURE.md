# 子问题学术进展与可迁移机制

检索/核验：2026-10-30。只采用论文、作者主页、官方仓库。这里的适配判断是研究推断；论文报告不代表本项目复现。检索覆盖结构控制、初始化、选图、深度可信度、优化加速；不声称穷尽所有工作。

## 1. 结构控制：从“哪里梯度大”转向“哪里表示带宽不足”

**SAD-GS，SIGGRAPH 2026**，arXiv v1 2026-04-30：[论文](https://arxiv.org/html/2604.28016v1)、[作者项目页](https://vcai.mpi-inf.mpg.de/projects/SAD-GS/)、[公开代码](https://github.com/LinjieLyu/SADGS)。它将训练图的多尺度结构张量/局部频率与投影高斯尺寸比较，经多视角统计后沿需要的轴分裂，减少逐轮盲目细分。作者页报告相对 FastGS-Big 2.8 倍提速和 LPIPS 降低 0.017，均是其自身协议结果。

适配：这是本轮最值得优先验证的 B 路线。只迁移“分裂对象/方向的选择”，保留现有 FastGS、短程日程与官方最终渲染。原实现依赖扩展 rasterizer，且 README 明示不同数据集使用不同超参数；不可把整仓替换或其 53 秒结果视为项目收益。轻量移植必须标记为受启发方案。详见 E1。

**Efficient Confidence-Aware Densification for 3DGS**，2026-10-09 预印本：[全文](https://arxiv.org/html/2610.13541v1)。它指出 opacity 采样偏向已解释前景，以误差和逐高斯置信度选择放置位置，以最大混合贡献选择被迁移者；两种改变需配合。其 confidence 是场景内优化参数，并不等于要训练额外跨场景网络。全文实验约分钟量级，使用 Faster-GS；本文未核验到作者代码发布。

适配：可测试“低贡献让位、高误差获容量”，不复活完整 MCMC，也不把 opacity 当 uncertainty。这里只迁移免新增网络的机制，称为本地近似，不称论文复现。详见 E4。

**Improving Adaptive Density Control，VISAPP 2025**：[论文](https://arxiv.org/html/2503.14274v1)。其修正场景范围、递增梯度阈值和贡献感知剪枝，说明仅用相机中心定义 extent、再按世界尺度剪枝可能伤害背景。适配重点是检查 `scene.cameras_extent` 与已有背景几何的关系，为 E4 提供背景保护对照；不是再扫一轮 opacity 阈值。

## 2. 初始化：密度与正确的空间位置必须区分

**EDGS，CVPR 2026**：[作者页](https://compvis.github.io/EDGS/)、[代码](https://github.com/CompVis/EDGS)、[论文](https://arxiv.org/abs/2504.13204)。通过训练图像对的稠密对应三角化，一次性提供几何、颜色和尺度，以减少渐进 densification。它支持“用正确支撑缩短优化路径”，不支持“把原高斯复制一倍就会改善”。

本项目不建议直接替代 ReSplat，或先加入昂贵匹配模型；前者改变固定链路，后者有初始化成本。只有当单次 ReSplat 已证明某类区域缺支撑，才考虑利用已有训练 SfM 轨迹作极小补充，且另立实验、计算全部耗时。

**The Role of Initialization in 3D Gaussian Splatting，ACCV 2026**：[2026-10-07 v4](https://arxiv.org/abs/2603.20714)、[代码](https://github.com/deivse/ivd_splat)。系统比较表明，强 densification 下，稠密初始化不稳定带来 NVS 提升，几何和离轨视角改善更明确。因此旧文档中“初始化很重要 → 任何初始化整理都是最大杠杆”的推理过强；本地随机初始化崩分只能证明 ReSplat 有价值，不能证明任意清理有收益。

## 3. 选图：真实表面覆盖比相机包围盒更贴题

**Z-Order Transformer，CVPR 2026 Oral**：[论文 §3.4](https://arxiv.org/html/2605.13465v1)。其最大覆盖选图依赖每个视角的 point map，贪心增加可覆盖的三维内容。它不是只按 camera-center FPS，也不是现有 coverage 的虚构包围盒体素。

适配：从已有训练SfM观测构建饱和覆盖和共视图，单次选满16张；不能为选图再跑额外ReSplat，也不把稀疏tracks等同原论文point maps。07:34已归档bicycle三策略负结果，且从images.bin共享ID解锁真实训练共视；下一步见[E2-T](TRACK_ONLY_SELECTION_20261030.md)。低阈值下三组都连通，仅加连通不足，应先CPU检验保单覆盖/增双覆盖的不同集合。

## 4. 深度：监督哪里、相信哪里比单纯增加 loss 更关键

**In Depth We Trust，CVPR 3DMV Workshop 2026**：[论文](https://arxiv.org/html/2604.05715v1)。使用深度不一致区域掩码和梯度对齐，避免把有噪先验传播到已正确几何。其原方案用虚拟双目额外渲染；完整移植有运行成本。本轮核验到论文，未确认可直接安装的作者代码。

适配：分开构造“先验可信”与“当前模型需修复”两个掩码，监督交集。使用同一次 ReSplat 的 depth sidecar，不追加新深度网络；尺度、camera-Z/射线距离及 crop 必须先校准。详见 E3。

**DRGSplat，作者页列 BMVC 2026**：[Google Research](https://research.google/pubs/drgsplat-depth-regularized-3d-gaussian-splatting/)。包含深度、法向和不确定性感知平坦项，侧重几何；页面仍写代码将发布。它提醒我们几何提升不等价于 VGG-LPIPS 提升，因此本项目先不堆法向/曲率项，也不引入额外先验。

## 5. 优化加速：残差驱动稀疏计算有潜力，但迁移层次要讲清楚

**TurboGS，ICML 2026（作者页/论文标注）**：[论文](https://arxiv.org/abs/2606.15924)、[项目页](https://zhengdong.site/projects/TurboGS/)。基于误差与 age 的 tile/pixel 采样、稀疏结构损失、密度控制和混合优化器；其 100 秒结果在 RTX 5090。本轮打开的项目页未给可用代码入口。

可先试只调训练 camera 采样的 E5；这只是借“把计算给未解释区域”的思想，不是 TurboGS 复现。完整渲染后再乘 mask 并不会节省 rasterization 成本；要获得论文像素稀疏化收益，需要专门 kernel，暂不作为第一优先。

## 6. 纠正一条旧文献归类

`research/LIT.md` 把 **Taming-3DGS** 写成“MCMC + regularizers”，不准确。[原论文](https://arxiv.org/abs/2406.15643)采用有预算、分数引导的构造式 densification，并优化梯度和属性更新实现。它与 [3DGS-MCMC](https://arxiv.org/abs/2404.09591)应分开。不能因为本地 MCMC 超预算就把所有有预算的结构方法一并排除。

## 本轮适配结论

### 06:35新增：Faster-GS与AA-Splat

[Faster-GS，CVPR2026](https://arxiv.org/html/2602.09999v1)，[作者代码](https://github.com/nerficg-project/faster-gaussian-splatting)整合高效backward、kernel融合和更新实现，是不同于当前FastGS的系统。先抽出fused dense Adam作为E8-F，不直接换整个训练器；Morton重排的收益依赖其per-Gaussian backward，不能作为当前FastGS的独立免费优化。完整倍数对比不是本项目实测。详细边界见 [优化器方案](SPARSE_OPTIMIZER_20261030.md)。

[AA-Splat，2026-03预印本](https://arxiv.org/html/2603.29394v1)，[作者页](https://kaist-viclab.github.io/aasplat-site/)研究前馈高斯的带宽与opacity平衡。它重新训练DepthSplat，并用渲染时逐像素alpha归一化；并非对固定ReSplat做一个PLY后处理就可复现。作者页当前仍写代码将发布。论文大幅收益主要在改变采样率的设置，本项目固定处理分辨率且官方renderer不做其颜色归一化，暂不作为新训练路线。

3D平滑后的scale/opacity可以表达在标准PLY中，但视角/像素相关的alpha颜色归一化一般不能静态烘焙成一套SH系数；只搬平滑而丢掉opacity处理也可能增加遮挡。后续若训练视角诊断证明有退化footprint，再设计保守输出过滤对照；不先重训网络或替换官方评测。

一个本地解析边界：同一视角的单高斯颜色在各覆盖像素相同；若opacity=.5、两个像素的Gaussian权重为1和.5，alpha归一化都要输出颜色c，标准合成则需同时把该高斯颜色设成2c和4c，无法由同一个静态SH颜色实现。这说明完整OB一般不能靠一次PLY颜色重写等价实现，并不否定可单独表达的3D平滑部分。

|方向|论文成熟度|本项目下一步|
|---|---|---|
|结构/频率感知分裂|有论文、有作者代码|E1 优先，轻量机制对照|
|训练观测覆盖与共视支撑|选图机制有论文，真实ID已核验|E2-T先CPU候选，再单次16图完整对照|
|选择性深度正则|有完整论文|E3，先做先验及深度渲染审计|
|贡献与误差联合容量重分配|2026-10 新预印本|E4 备选，不承诺短跑收益|
|误差与 age 的训练视角调度|相关像素级论文已发表|E5 低成本备选，明确为本地假设|
|新学习型前馈模型/生成式纹理|多工作存在，但改变网络/资产或增加依赖|当前不排期|
