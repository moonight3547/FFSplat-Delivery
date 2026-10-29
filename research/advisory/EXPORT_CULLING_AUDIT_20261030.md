# 导出可见性审查：球形距离剔除不等于相机far-plane

2026-10-30 07:29 +08:00。源码/环境只读审查与CPU几何反例；未修改运行代码、未做GPU实验。它是当前导出启发式的语义边界，**不是已经证明的真实场景质量故障**，也不与SH符号错误混为一项修复。

## 观察到的实际代码

固定`baseline/run_resplat.py:317`按所有context的最近**欧氏距离**保留高斯：

```
min_i ||mean_world - camera_center_i|| <= far
and opacity >= 0.005
```

注释把它解释为去掉ReSplat原生far-plane永远看不到的离群高斯。但原生decoder传给gsplat的是near/far平面，不是以相机为球心的距离。

[H1已安装包只读证据](evidence/20261030-0730-gsplat-far-semantics.json)发现ffsplat-enhance环境的gsplat1.5.3。其`ProjectionEWA3DGSPacked.cu`将世界高斯变到相机空间后检查`mean_c.z`，Python参考实现也取`means_c[...,2]`再比较near/far。文件路径与hash已归档；[gsplat官方参考实现](https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_torch_impl.py)语义相同。安装目录有python版本别名，归档保留所见全部路径；没有导入模型或分配GPU。

## 独立几何反例

16个相机中心沿x从−1到1、朝向+z，图像960×576，fx=fy=480，主点(480,288)。点(150,0,180)在每张图中心视锥内：u约877–883，v=288，camera-Z=180<far200。但它到最近相机的欧氏距离约233.67>200，当前导出会删除。

[标量记录](evidence/20261030-export-far-scalar-counterexample.json)证明两种判据不等价。它只核验中心/相机平面与投影，不是CUDA像素贡献测量，不证明真实ReSplat恰好生成这些点。

反过来，单个全局keep集合也无法在每个相机下精确重现far-plane：某点在A相机z<far、B相机z>far，静态PLY保留后，缺少对应far裁剪的后端仍可能在B看到它。因此把距离改成“任一context的z有效”也只是另一种启发式，不能称为两种renderer的精确等价修复。

## 当前bicycle证据的上限

[已有三组日志](evidence/20261030-0718-selection-native-render-logs.json)显示每次网络原生552,960个高斯，导出分别删31,980 / 32,503 / 32,686（约5.8–5.9%）。日志把far与opacity合并计数，不能知道有多少因为球形far、多少因为低opacity、多少同时命中；更不能从删除数量推断图像贡献。

所以native context自渲染与导出PLY比较还混入了**高斯集合不同**，除了已有的SH旋转和光栅化差异。不能把所有差异都归给其中一项。已经导出的PLY/NPZ若只保存keep后数据，也无法从它们重建被删集合。

## 给coding agent的最小诊断包（归入E11-D0）

在下一次本来就要运行的合法单次前馈中，保留未过滤张量或只保存以下统计；不为择优再做额外前馈，不改固定网络/权重/refine。

1. 分开统计radial-only、opacity-only、两者同时命中、保留四组；按预测来源context、深度段、opacity段记录数量与scale。
2. 对radial-only组，在原context相机计算camera-Z及投影中心/协方差足迹候选可见性；明确这仍不是混合后的真实贡献。
3. 若候选可见量非零且值得继续，使用同一次预测，在同一gsplat后端重渲染删前/删后context，比较RGB/alpha差异和训练误差。只渲染、不追加网络推理；全部诊断成本记录。

如果radial-only几乎没有，或删前/删后几乎无图像差异，这条路线就关闭。若主要为opacity组，不调整far；转入带贡献保护的E4诊断，避免盲目降低阈值保留噪声。

只有实际贡献与质量问题同时命中后，才讨论一个统一的保守keep规则对照。保持SH修正状态一致、opacity阈值不同时改变，比较B=0及5K最终质量，并计入额外N带来的每步成本。首先诊断现有规则，不先移除全部culling或扩大far到极大数。

这项审查可以解释某些远背景/边缘内容的导出落差，但目前没有证据说明它是两个LPIPS零分case的主因。
