# FFSplat-Delivery 交接（2026-10-31）

主仓：https://github.com/moonight3547/FFSplat-Delivery （private）。源 `45feb68` 保存在 `deepseek-testing-45feb68`；实际目录快照 `d7a3754` 保存在历史。用户确认 DeepSeek 后续不再修改仓库，只由用户转达建议；Codex 负责实验裁决及最终 setting。原测试版本仍保存在历史，开发在独立 worktree 进行。

用户已启用默认 **BigD3500 + VGG200**；confirmed 模式使用 r0、5000 步匹配 LR、foreach Adam、FP32、关闭 TF32、修正 SH 导出和无损 PLY。README 和 configs/confirmed_recipe.json 是当前入口；DELIVERY.md 的旧 clean13 是 legacy 证据，不能用于新配方成绩声明。

用户最新明确：**不再尝试 D1250，改为 D4000**。新确认实验为 D3500/D4000 × r0/r15@1250，固定 VGG200/.1，3 badcase × 2 seeds，共 24 个 fresh models；D1250/r15 只保留历史，不新跑。用户未要求把 D4000 设为默认。已有 fresh D4000/V200 双种子对照综合分较 D3500 低约 .263，因此仍保持 D3500。

训练源代码须本地修改并 Git 提交，远端 Git 获取；不通过 SCP/stdin 发源码。管理凭据只在控制端 ignored/private 或系统凭据中保存。新实验独立输出，保护唯一证据；不得抢占合作者 N1/H1。现有研究 sub-agent 继续负责 H3 参数/SH、N2 Lite/重分配、N3 尺度/absgrad；资源安排与原始证据仍在研究仓，不把旧远端地址当成当前授权。

验收范围、待办与 CPU/GPU 验证见 [DELIVERY-REVIEW-20261031.md](DELIVERY-REVIEW-20261031.md)。全 13 场景及新完整 B 时间范围需要另外实测，当前没有已确认的新最终得分。迁移前交接见 [历史文件](HANDOFF-20261029-HISTORICAL.md)，其 E2E 主门槛、旧机器和未启动状态均已过时。

本轮验证：69项CPU通过；H3/GPU2 source407505c完成600步/VGG200、native600与完整25TEST官方重载渲染。顶层baseline/resplat已按官方cb29814组织，最终ZIP需通过build_submission.py从当前已提交HEAD导出并保留SUBMISSION.json。旧DeepSeek后续13文件简化另存7734515，不静默覆盖，尚无组合证据的r15/auto剪枝不默认启用。
