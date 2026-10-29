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

## 06:46：在真实prepared元数据上运行既有CPU选图器

原始证据：[CPU选图诊断](evidence/20261030-0646-selection-cpu-audit.json)。使用新增节点现有select_views模块，source hash与本地相同；只读相机/split，未读取RGB、未调用ReSplat或GPU，远端不写输出。

|case/policy|与uniform重合张数|实际pivot|SH一阶算子差异范数|
|---|---:|---|---:|
|bicycle uniform|16|_DSC8782.png|2.7638|
|bicycle fps|1|_DSC8767.png|2.1189|
|bicycle coverage|1|_DSC8743.png|1.7083|
|hw uniform|16|248773015215078.png|0.4313|
|hw fps|6|248773015215078.png|0.4313|
|hw coverage|5|248767981376078.png|0.2402|
|counter uniform|16|DSCF5984.png|2.5397|
|counter fps|3|DSCF5970.png|2.8268|
|counter coverage|3|DSCF5964.png|2.8282|

uniform在这三份prepared输入上与冻结context完全相同；其他策略确实改变集合。CPU纯选择约0.002–0.14秒，不包括conda启动、复制、ReSplat和其他A成本。coverage报告的proxy weighted fraction约bicycle0.7428、hw0.9462、counter0.8789，仍不是表面/图像实际覆盖率。

**重要混杂**：保持context[8]索引不等于物理pivot不变。bicycle两种候选同时降低了已发现SH导出错误的算子差异；若它们质量更好，修复前的A/B无法把收益全部归为选图。counter候选的算子差异反而略大，也不能由此预测质量，因为实际SH能量/网络输出同时变化。优先对所有策略统一修正SH导出再比较，不能按场景挑一个更有利的pivot来绕过问题。

**语义边界**：当前coverage只读pose文件，代理盒来自相机云及平均光轴，稀有视锥体素获得较高权重，没有表面/遮挡/真实tracks。FPS和coverage的负结果只能裁决这些代理；它们的正结果也不证明真实共视目标最优。selection_hard入口称hard/init-limited只是实验命名，不能把待查病因当已证明事实。

一个后续轻量方向是对潜在重叠/连通增加约束，但没有tracks时仍只能叫pose-proxy；如需加入，先以此CPU诊断观察孤立视图/近零增益槽位，不立即另开大网格。真正E2仍优先复用原始观测图。

## 06:54：真实tracks已在hw找到，E2部分解锁

[文件库存](evidence/20261030-0652-track-inventory.json)发现既有billboard COLMAP临时目录保留points3D.bin；进一步用其 **undist/sparse** 完成只读[轨迹统计](evidence/20261030-0654-hw-tracks-audit.json)，不新跑SfM、不用RGB。源文件header/完整解析与hash已保存；64张source图全部映射到prepared，统计只使用56张训练图，N1选择输入与H1 split的hash一致。

共有20,404点，其中20,175点有至少两张训练图观测。使用已有三种16图集合：

|策略|至少一次训练轨迹覆盖|至少两次训练轨迹覆盖|共享≥20轨迹的连通分量|
|---|---:|---:|---:|
|uniform|84.01%|45.02%|1|
|fps|83.84%|43.88%|1|
|coverage|80.49%|45.01%|1|

所有选中视图对至少共享25条以上轨迹；本例没有选集碎裂。故**只增加连通约束不是本例最有证据的改进机制**。当前proxy coverage的高代理分也没有变成更高真实轨迹覆盖，支持改用真实观测目标进行一次小对照，而非继续调空体素margin。它尚未证明最终图像质量会改善。

可实验的E2首版应聚焦饱和轨迹覆盖/双视角支持及角度，连通仅作保底；优先不增加新的匹配网络或SfM。84%/45%仍是稀疏纹理特征点覆盖，不是全表面覆盖，不能把剩余比例全当成可恢复的质量空间。

原始SfM由全部输入构建，继承A阶段all-view pose来源；本统计仅用训练image IDs，没有使用point位置、颜色或全模型重投影误差作为选择权重。正式方案也不应按test观测数或含test的误差排序；需要几何权重/坐标时，优先从训练观测和已允许的相机做训练侧重投影/三角化检查，并记录继承的pose来源。

目前同一H1路径下counter/bonsai原始sparse目录缺失；bicycle有images.bin但没有points3D.bin；hw prepared本身仍没有tracks。不能把billboard的可用性推广为全部场景。统一回退规则继续有效。

另查 [POp-GS，CVPR2025](https://arxiv.org/html/2503.07819v1)：它利用已有GS模型及渲染导数估计信息，不是仅有pose即可得到的免费A侧分数。不能为A先做一次ReSplat再据此重选并做第二次；若借信息准则，必须从本来已有的训练侧SfM观测另行定义，并明确不是POp-GS复现。
