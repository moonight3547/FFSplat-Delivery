# 学术调研与实验建议：coding agent 固定入口

更新：2026-10-30（Asia/Shanghai）。保存调研、实验方案、问题建议和只读证据；研究 agent 不实现训练算法、不启动训练。用户本轮明确追加授权相机可视化脚本与13例本地产物，该项由专门subagent实现。

**12小时研究进行中：05:43:57–17:43:57 +08:00。** 进度见 [连续研究记录](AUTO_RESEARCH.md)。下一步先读 [实验决策板](DECISION_BOARD.md)，不要同时实现所有备选。

**13例相机轨迹已生成**：[本地离线3D网页](visualizations/cameras-20261030/index.html) · [俯视总览](visualizations/cameras-20261030/topdown_overview.png) · [脚本用法](../VISUALIZE_CAMERAS.md)。包含3,009个真实相机，两个已知LPIPS零分case标红，三个缺失指标标未知；4份冻结/9份重建划分均已说明。Chrome逐个加载13场景通过，无JS错误、无外网请求；四策略和俯视按钮实测通过。

## 当前结论与优先级

- **先核查SH导出**：[SH接口审查与F0–F2](SH_EXPORT_AUDIT_20261030.md)。一阶z基函数符号与官方FastGS不一致，CPU反例和H1源码/pivot已核验；尚无真实场景最终增益。
- **E3先修梯度**：[深度接口审查](DEPTH_INTERFACE_AUDIT_20261030.md)。仅detach归一化分母会制造错误梯度；alpha也不等于预测可靠性。修正与零权重控制通过后才做可靠先验×待修复区域。
- **当前实验状态**：[同步07](SYNC_20261030_07.md)。bicycle三选图完成，uniform/FPS/coverage LPIPS为.2982/.3143/.3112，均0分且后两更差；只裁决这些代理，不关闭整个A。13例真实相机位姿已收齐，质量仍缺3例。
- **优化器路线分开裁决**：[E8/E8-L/E8-F](SPARSE_OPTIMIZER_20261030.md)。降频有负结果叙述；可见性稀疏未完成，fused dense Adam接口已核验，性能待测。
- **当前控制**：raw ReSplat + cache + FastGS5K、LR长度5K、无reset、DSSIM .6、scaling LR .01、结构截止1250。历史sharp/无cache结果单独保留；首次cache成本计入E2E。

## 研究与实验索引

|内容|文件|
|---|---|
|执行顺序、状态与回执约定|[决策板](DECISION_BOARD.md)|
|大颗粒方案E0–E5、E10及共同协议|[实验方案](EXPERIMENTS.md)|
|一手学术来源与适配筛选|[文献](LITERATURE.md)|
|源码问题及修改意见|[coding建议](CODING_NOTES.md)|
|E1结构感知分裂的诊断与等数量对照|[结构迁移](STRUCTURE_TRANSFER_20261030.md)|
|E2真实轨迹与选图前置条件|[选图可行性](VIEW_SELECTION_FEASIBILITY_20261030.md)|
|E2-T：无points3D坐标文件也可用的真实ID共视|[新共视候选](TRACK_ONLY_SELECTION_20261030.md)|
|E3深度定义/梯度与可靠先验|[深度接口审查](DEPTH_INTERFACE_AUDIT_20261030.md)|
|E3-S已有训练轨迹的稀疏几何锚点|[几何锚点方案](SPARSE_GEOMETRY_ANCHORS_20261030.md)|
|E11零分难例的接口/覆盖/欠拟合分层诊断|[难例归因](ZERO_SCORE_DIAGNOSIS_20261030.md)|
|E4/E7与已有FastGS机制的差别|[机制审查](FASTGS_MECHANISM_AUDIT_20261030.md)|
|E6局部单步约束、二阶路线成本|[优化器文献筛选](OPTIMIZER_OPTIONS_20261030.md)|
|E8可见性/降频/fused更新|[优化器实验](SPARSE_OPTIMIZER_20261030.md)|
|E5-G按视角跳过冗余backward|[反传门控](BACKWARD_GATING_20261030.md)|
|E9固定几何后的颜色/SH求解|[外观求解](APPEARANCE_SOLVER_20261030.md)|
|ISP/神经外观的PLY边界，E9-P静态gain诊断|[外观兼容性](APPEARANCE_COMPATIBILITY_20261030.md)|
|E0缓存首次构建与计时边界|[缓存审查](CACHE_TIME_BOUNDARY_20261030.md)|
|raw/sharp变换与低opacity分布|[不透明度审计](OPACITY_AUDIT_20261030.md)|
|导出球形距离far与原生camera-Z裁剪的区别|[可见性审查](EXPORT_CULLING_AUDIT_20261030.md)|
|历史控制与评分缺口|[初始状态](CURRENT_STATE.md)|
|飞书连接结果|[飞书](FEISHU.md)|

## 证据与同步

最新：[同步10](SYNC_20261030_10.md)。历史：[01](SYNC_20261030_01.md)、[02](SYNC_20261030_02.md)、[03](SYNC_20261030_03.md)、[04](SYNC_20261030_04.md)、[05](SYNC_20261030_05.md)、[06](SYNC_20261030_06.md)、[07](SYNC_20261030_07.md)、[08](SYNC_20261030_08.md)、[09](SYNC_20261030_09.md)。不可变原始JSON位于 [evidence](evidence/)；文件名为采集批次，精确时间以内容的captured字段和同步说明为准。

完成实验后请另写结果回执到 `results/<日期>-<方案ID>.md`，链接原始记录并给出具体裁决；没有运行就明确写未运行。不要覆盖旧证据，或仅把方案正文改成已验证。

## 固定约束

单次最多16个训练context送入固定ReSplat，权重/网络不变，num_refine=2；test RGB不进入训练、选图、停止或逐例超参。参考为同GPU UUID、同处理输入的随机FastGS真实30K；默认三例与13例分别报告。

依ProblemDesc同时保留A/R/B/E2E及B-only诊断，官方总分未确认。估算时间不改名为实测；失败/超预算case不从完整集合中删掉后报有效均值。所有候选最终使用官方完整tile渲染和VGG-LPIPS评估。
