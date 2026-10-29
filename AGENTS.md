# 新对话入口

先读 README.md、ProblemDesc.md、docs/HANDOFF.md。此仓库是干净的新主入口；旧Enhance/Eval只作历史证据，不拉入mixed10代码，不恢复旧队列。

- 用户确认的预算：同GPU、同场景、同处理图像，随机初始化FastGS30000实测。参考目录reference_fastgs；禁止经典3DGS预算或ReSplat接续30K冒充。赛方原文仍有差异，明确标注。
- 官方代码原样存于competition/upstream，cb29814快照；校验upstream-snapshot.json。固定ReSplat网络/权重/num_refine=2，不用测试RGB指导A/B，也不做逐场景超参选择。
- 默认3case，可选13/confirmation2不自动扩展。指标PSNR/SSIM/VGG-LPIPS；展示A/R/B/E2E秒数、预算、分项代理分和有效性；官方总分未确认。GPU UUID不同，不合并为同机结果。
- 保持原始证据不覆盖。新实验使用全新work目录，原控制组只允许重用经验证的几何/输入；计时必须匹配当前设备。
- Conda和profile管理环境。Mac/Windows控制Linux GPU。不占用合作者备用GPU，除非必要且确认空闲。
- 所有API token、SSH密码/私钥只保存在本地ignored private/或系统凭据。不要提交，不写URL，不传管理凭据到GPU。源代码本地修改、GitHub提交、远端Git获取；不通过SCP/终端stdin发送项目源代码。
- 优先先处理docs/HANDOFF.md所列标定，再重新规划超越baseline。不得把历史正分当作当前口径得分。
- 每轮维护README及交接记录；用户另有新指令时更新实际状态，不沿用过时的“停止/待确认”段落。
- 不推断已授权新费用或抢占资源。不需要回到旧仓库完成历史上传任务，资产共享已完成。
