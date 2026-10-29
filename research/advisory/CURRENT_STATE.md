# 当前进展与证据边界

核查日期：2026-10-30。读取项目 AGENTS、ProblemDesc、HANDOFF、研究文档、SUMMARY、配方 JSON、交付三例原始摘要和关键研究脚本。下述仅描述本地证据，未访问 GPU 当前队列。

## 已建立的工作基础

固定 ReSplat 初始化，FastGS 短程优化已经比默认 10K 调度更合算。当前统一交付配方为：一次 16 context 前馈 → opacity logit 放大 2 倍 → FastGS 5K，位置 LR 日程 5K，关闭 opacity reset，DSSIM 权重 0.6，scaling LR 0.01，结构操作截止 1250 步。

来源：`research/experiments/delivery_check.sh`；原始记录在 `records/artifacts-20261030b/h1-delivery/*/summary.json`。

|场景|PSNR|SSIM|VGG-LPIPS|摘要 B 估算秒|train_record.total_seconds|摘要单场景代理分|
|---|---:|---:|---:|---:|---:|---:|
|DL3DV 001dccbc…|34.268|0.9667|0.1107|34.28|39.044|77.4184|
|hw/billboard|22.780|0.8083|0.1426|24.63|30.387|53.0878|
|mipnerf360/counter|27.359|0.8916|0.1994|31.05|36.203|50.1032|

摘要单场景分算术均值为 **60.2031**。这只是原记录的 **B-only、快照时间估算、逐场景分均值**，不是当前 E2E 的验收分。

另有两个独立问题：

- 快照 B 时间是 load + init + snapshot train + 固定 0.5 秒，没有真实最终导出计时。`train_record.total_seconds` 又包含最终 quick test 和重复导出，不能直接拿它当正确替代分子。
- `ProblemDesc.md` 要求先平均质量指标再映射分数。三例均值为 PSNR 28.135667、SSIM 0.8888667、LPIPS 0.1509，对应 Q=45.437867；配原有估算 TIME 均值 15.210607，得 **60.648474**。它与 60.2031 差 0.445333，原因是逐场景 SSIM 截断不与平均交换。这个数字仍不是验收分，只用于定位聚合差异。

默认预算在后续记录为 94.29/99.49/92.57 秒，早期 HANDOFF 为 102.68/100.13/93.72 秒。不要按“日期新”直接替换预算；应检查 GPU UUID、输入身份、参考训练器和计时边界后配对。

## 已有反例：不要重新包装成新路线

|方向|现有证据及边界|
|---|---|
|更长训练、普通 MCMC/经典 3DGS 接续|FastGS 约 100 秒预算下收益/时间不足；旧宽预算正分不能迁移|
|0.5→0.75→1 分辨率渐进|旧试验质量和总分差；不作为默认新提案|
|每步 VGG-LPIPS loss|已测成本高、质量亦未赚回；不能因为评分占 40 分就重推|
|盲目增加致密化/放宽门槛|已失败；不等于“使用不同空间分配准则”也被证伪|
|固定高斯数量，关闭结构操作|均值没有统一胜出；但这不是冻结几何的 appearance-only 对照|
|按 opacity / 世界尺度做预剪|均值下降；不等于多视角贡献保护的预算分配已失败|
|更多 ReSplat refine|2 轮后饱和，而且固定协议为 2，关闭|
|训练后期 translation-only pose 微调|研究文档 §13 报告中性偏负；本地没有相应完整归档，记为叙述级反例，不重复跑同配方|
|多子集 ReSplat 拼接|最多 16 帧一次前馈约束明确排除|

主要证据：`records/SUMMARY.md`、`records/best-recipe.json`、`docs/RESEARCH-20261030.md` §12–14。SUMMARY 的 159 行统计不含后来归档的 h1-delivery，不是最新所有实验的全集。

## 只有入口或局部记录，尚不能宣称结论

- `research/experiments/selection_test.sh`：uniform/fps/coverage 的完整单次前馈 + 当前交付 B 对比入口存在。本地有旧 round4 部分选图 JSON，但没有该完整对比结果。
- `research/experiments/eval13_recipe.sh`：HEAD 新增的是 13 场景执行脚本，本地未找到 13 例完整归档；不能说已测 13 例。
- `research/depth_reg.py`、`export_resplat_depth.py`：有实现与文档，未见本轮深度正则成套结果。既有接口不等于有效先验已验证。
- `research/ply_ops.py --op merge`：实际是 PLY 拼接，不是去重或矩匹配合并；不能引用为“已实现冗余合并”。

## 对当前问题的判断

瓶颈同时包括弱初始化的局部几何/纹理、有限训练时间下的空间容量分配、以及证据口径不一致。hw 的低分不能仅凭 PSNR 平台就诊断为 pose 错误；要同时观察训练残差、共视支持、深度一致性和输入/坐标审计。后续优先找改变误差空间分布的机制，避免继续只调训练长度。
