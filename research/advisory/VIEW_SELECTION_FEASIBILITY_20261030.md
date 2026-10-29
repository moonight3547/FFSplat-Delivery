# E2 可行性审查：先确认观测证据，再谈共视优化

日期：2026-10-30。本轮读取赛方固定源码与最新公开作者资料；未实现新选图器。

## 关键前提被收窄

`competition/upstream/baseline/module_a/preprocess.py::write_work_dir` 导出的 `images.txt` 每张图片只有 pose 行，下一行为空，没有二维观测或 point3D IDs。它也不在这个 prepared work 中写出真实 `points3D` tracks。因此“现有 prepared work 是COLMAP格式”不等于“已有SfM共视图”。

原数据可能保留完整 COLMAP（如 mipnerf360），hw 的既有 COLMAP 临时目录也可能包含 tracks；但 transforms.json 来源的 DL3DV 不天然有这些观测。本轮没有为了 E2 新跑 SfM/匹配，也没有验证这些原始轨迹文件完整存在。

**对 E2 的修订**：先列原始 source/已有COLMAP临时目录的可用轨迹，不从 prepared pose 文件伪造观测。若只有部分场景可用，采用预注册的“有足够真实tracks则用graph，否则uniform”统一规则，并报告触发率；不按场景标签人工选策略。若必须额外重跑SfM，需先计算其A/E2E成本，这条路线立即降低优先级。

## 真实共视、潜在重叠和覆盖代理是三件事

- 真实共视：同一个可靠三维轨迹点被多张训练图实际观测到，有证据支持建立边。
- 潜在重叠：相机视锥/光轴/基线判断某些区域可能共同可见，尚未确认表面/遮挡。
- 当前 coverage：用相机云包围盒构造代理体素，最多只能衡量代理覆盖，不能称实际表面覆盖。

没有 tracks 时可以研究更好的位姿代理，但必须更名、降置信度，不把它报告成图像匹配支撑的共视。

## 文献借鉴的范围

[CoMapGS，CVPR2025 作者页](https://youngkyoonjang.github.io/projects/comapgs/)使用共视图处理欠观测区域，[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_CoMapGS_Covisibility_Map-based_Gaussian_Splatting_for_Sparse_Novel_View_Synthesis_CVPR_2025_paper.pdf)的图依赖稠密对应，不是靠相机中心直接得到。不能把其效果归给零成本 pose-only 代理。引入其新匹配模型和初始化流程也不是本项目首选。

[ReSplat 作者页](https://haofeixu.github.io/resplat/)展示多视角数和跨数据泛化；本地固定源码有 bounded window sampler，且额外视角可用 farthest_point。由此不能武断说“FPS违背ReSplat训练分布”或“均匀选图是模型唯一正确输入”。配置默认视角数也不能证明当前16视角权重的完整训练细节。真正裁决仍是冻结权重下的单次16图 + 同B完整对照。

## 后续最小任务

1. 读取已运行 selection 的三组完整结果；如果输出尚在远端，先同步，不重复训练。
2. 为真实tracks建立 provenance：source路径/hash、图像名称映射、resize/crop映射、仅训练观测的过滤计数、每轨迹训练支持数。
3. 目标函数先用可靠表面点的饱和覆盖，加连通约束；不要只最大化baseline或奖励从未被验证的空体素。
4. 相同16图、相同排序约定；记录实际 `context[8]` pivot 的图像名与相机。不同集合可能改变物理pivot，不应把同一个索引理解为同一个物理参考相机。
5. B=0只诊断，最终质量和A新增耗时共同决定收益。未获得新表面信息时，选择策略没有理论保证胜过uniform。

## 本轮另一项排除性检查：主点

FastGS加载器只保留focal/FoV，构造居中投影；ReSplat读取完整K。为检查是否解释当前弱结果，本轮只读提取了主机已有9份 `cameras.txt`：[统计](evidence/20261030-principal-point-audit.json)。包括默认三例和当前部分eval13场景，所有 `cx-W/2`、`cy-H/2` 均为0。

**结论**：代码支持范围确有差别，但当前已检查的这些输入没有主点偏移，所以此假设不能解释它们的低分；不建议为这些case修改投影矩阵。没有据此验证其他内参、畸变或所有13例，只排除本项。
