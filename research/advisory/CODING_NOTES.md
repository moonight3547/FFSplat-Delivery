# 给 coding agent 的修改意见

2026-10-30。本文件是代码阅读意见；研究 agent 未修改下述实现。优先级按对实验可信度的影响排序。

## P0：eval13 清理新增了源数据删除

**状态更新（12小时研究首轮，2026-10-30 05:44）**：coding agent 已在 `be4537a` 移除对 `$src` 的删除，且加入结果存在检查。以下段落保留历史问题及建议，不再称本地最新脚本仍有这个删除缺陷。远端是否运行新版本需按实际脚本摘要核对。

写作期间另一工作流新增提交 `30ff13b`，`research/experiments/eval13_recipe.sh` 末尾清理把 `"$d/work/images"` 和 `"$src"` 加入 `rm -rf`。`src="$DATA/$group/$scene"`，即输入数据集场景目录，而非本次输出的临时目录。当前脚本也不保证先检查 summary/metrics 复制成功才清理，训练失败可能仍走到删除。

建议 coding agent 继续执行前核查：常规清理只允许本次 work/output 下的生成产物；若磁盘确需逐场景下载并移除缓存，显式区分 task-owned download cache 与用户原始 DATA，只有当前任务拥有且可重建的缓存才能清理，并在结果/身份/最终资产成功归档后进行。保留最终交付 PLY 或有可追溯的远端资产；初始 PLY 不能冒充最终 PLY。研究 agent 未执行任何删除，也未修改或停止另一工作流。

## P0：把候选计时与参考边界对齐

位置：`research/run_sweep.py:174`、`research/train_sweep.py:310`、`competition/upstream/baseline/module_b/train_from_ply_fastgs.py:181`。

当前 summary 的 `b_seconds_estimate=scene_load+init+snapshot_train+0.5`，最后一项是固定估计，缺少真实最终导出与初始化整理时间。snapshot 时间戳在其自身 save 前，早期快照的保存开销又会污染后续累计时间。train_sweep 的 `total_seconds` 在 quick_test_psnr 后截取，而上游参考在 scene.save 后、quick test 前截取；两者不能直接比较。研究入口还可能同时保存 snapshot 和 final，交付不应无意为重复导出付费。

建议在自己的 wrapper 层新增真实交付计时段：加载/初始化整理 → 必要优化与结构统计 → 唯一最终 PLY 写盘结束，CUDA 边界同步；evaluation 独立记录。快照记录保留 estimate 字段，不能改名成 measured。参考使用相同边界。进程冷启动另报。不要改受哈希保护的 upstream。

验收：一份最终 PLY 的交付 B 秒能从分项重建；重复保存/测试渲染没有偷偷进入一侧分子；A/R/准备成本也进 E2E。原始记录保留，不覆盖重打分。

## P0：聚合与有效性

位置：`research/experiments/delivery_check.sh` 的末尾均值，`research/run_sweep.py::proxy_scores`，`records/SUMMARY.md`。

目前不少展示平均逐 case 的 clipped 总分；`ProblemDesc.md` 约定先平均质量原指标，再 clip/map，TIME 逐case计算后平均。交付三例上两者为 60.2031 与 60.6485（都仍含旧估算 B）。SSIM 截断使其不同。请把聚合函数与范围写入 schema，而不只保留一个 mean_total。

`proxy_scores` 对超预算 case 仍输出质量+0 的数值 total；若调用者忽略 valid，会把无效结果带入排名。建议汇总明确返回集合级 invalid，单独保留诊断分；任何超预算/失败 case 不能删除后继续声称完整集合有效。B-only 有效不等于 E2E 有效。

## P1：纠正不透明度操作的名字和公式

位置：`research/ply_ops.py:70`；`docs/RESEARCH-20261030.md`、PREPOST 文档中的公式。

实际是 `l_new = gain * logit(clip(sigmoid(l_old)))`，即 `p_new=sigmoid(gain*logit(p_old))`。gain=2、p=0.2 时输出约 0.0588；不是 `p→2p` 后取 logit（会得到 0.4）。其效果是将 p<0.5 压低、p>0.5 推高，不是统一增亮或提高 opacity。

请把 init transform 公式、gain、输入与输出 PLY hash、操作耗时写入每条结果；旧 campaignH 仅 tag 含 sharpen，settings 不记录，无法靠参数字典复现。

## P1：不要把固定拓扑称为纯外观优化

位置：`research/train_sweep.py:275`、`:290` 和 Gaussian optimizer setup。

`--no_densify` 关闭结构操作/重置，没有冻结 xyz/scale/rotation；梯度仍更新几何属性。已有 campaignI 证明的是“固定拓扑”的均值不胜出，不能据此说“只拟外观”已经被排除。若以后做真正外观对照，需显式冻结位置/尺度/旋转，并记录冻结参数列表。本轮不要求立刻实现这个新分支。

## P1：检查 densify 的父子索引对应

位置：`research/ply_ops.py:76`。

`index=concatenate([arange(N)]*factor)` 是按 child block 排列，但 `child_id=tile(arange(factor),N)` 是按 parent 交错排列，二者不匹配。N=2、factor=2 时 index 为 `[0,1,0,1]`、child_id 为 `[0,1,0,1]`，父 0 两次都被当原体，父 1 两次都被当副本。可能使“一份原体+若干副本”语义不成立。

若启用此路径，请先在最小父集合检查每父恰有一个原体，并使索引顺序与 child_id 一致。结构感知分裂不能建立在这个错误配对上。默认交付不用此操作，当前观察不能说明交付受其影响。

## P1：深度接口尚需真实集成检查

位置：`research/export_resplat_depth.py::GPU_TEMPLATE`、`research/depth_reg.py`。

导出模板假设 `run_inference` 的第 4 个返回值含 `depth_fullres`，还依赖 context 顺序和深度坐标约定；CPU self-check 不能证明实际模型版本满足它。请验证真实输出字段、形状、camera-Z/射线距离、尺度、context 图像对应关系，旁路捕获同一次推理。若使用编码 RGB 的深度渲染，请和小型已知几何比较：alpha、深度、遮挡、背景均要合理，且常数项/颜色 clamp 不破坏数值。

深度来自预测头而非重渲染，仍与 ReSplat 错误相关；增加 loss 不构成独立正确监督。记录实际命中有 sidecar 的训练步数。

## P2：证据标签与安全的实验组织

- `merge` 实为 concat；为真正去重/矩匹配另起名字，禁止沿用多子集方案进入交付。
- HANDOFF、README 与 SUMMARY 的进展版本滞后；由 coding agent 下一轮正常维护时区分“脚本存在/已运行/已归档”。研究 agent 本轮只改 advisory 目录，保留用户原 README 改动。
- selection/eval13 脚本以 metrics 存在判断完成，不足以保证当前代码、输入和 recipe 相同。应有 run identity；新增实验用新目录，防止跳过旧结果或清掉正在运行目录。
- 以 GPU UUID 配对参考，不能因“同为满血4090”就把跨卡时间当同机标定。h2 节流记录继续仅作质量证据。
- test PSNR 可以作独立离线报告，不能参与候选预算分配/选择。正式比较不从 test 曲线挑各场景最佳 checkpoint。

## 文献描述修订建议

旧 LIT 将 Taming-3DGS 归入 MCMC 不准确；旧 PREPOST 把固定拓扑叫 attribute-only、把 concat 当 merge、把估算分当完整实验结论，也需在正常文档维护时更正。不要再宣称“2025–2026 没有预算相关研究”：已有多类 primitive/计算预算工作，尚未检索到完全相同的冻结 ReSplat + 约100秒后优化设置，只能陈述本轮检索范围。

## 12小时研究追加：首批 eval13 的两个新缺口

1. 主 GPU 日志中 `0f622ea…`、`2dad2d…` 两例的 A 阶段报 `OSError: image file is truncated`，且后续重试仍失败。先核对来源文件的大小/完整性和资产校验；不建议启用 PIL 的截断容忍来伪造完整输入，不跳过坏图后改变 split 仍称同场景对照。研究证据见 `evidence/20261030-0548-h1-errors.json`。
2. 13例初始输出只有候选，baseline 是后来才加的；若清理删掉 `work/images` 但保留 `split.json`，续跑仅检查 split 存在会错误地认为 prepared work 完整。应显式检查图像清单、hash/尺寸和需要的几何文件，重新构造等价输入后再补 baseline；baseline/candidate 的输入身份一致才能比较。

## 12小时研究追加：不要把条件性质量上界写成不可能性结论

FINAL-REPORT 中“75+ is not reachable / hard ceiling ~65”超出证据。把**当前质量固定、时间设零**只给出当前模型的条件上界，不能限制改变几何/损失/初始化后能达到的质量。历史若干方法的最大值也不是所有未来方法的理论上界。建议表述为“现有质量下，单靠节省时间不足以到75；需要新质量增益”，保留研究开放性。
