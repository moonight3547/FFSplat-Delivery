# 提交方案审查与验证

私仓 FFSplat-Delivery：源45feb68，目录导入d7a3754。DeepSeek 原测试快照与实际收到目录均保留；修改在独立 worktree 完成。

## 已修正

- 默认从 D1250/r15/VGG0 切换为用户启用的 BigD3500/VGG200/r0；legacy 入口保留原参数和训练后端。
- B 使用显式 foreach=True/fused=False、FP32/TF32off、原生 SH-rest 每16步累积。拒绝环境变量静默换成 Faster-GS Adam。
- 原 safe_state 会把 requested seed 重置为0；现在在其后恢复 seed。VGG 构造保存/恢复 Python、NumPy、CPU/CUDA RNG，冻结权重，记录 backbone/linear SHA256。
- 接入已验证无损 vectorized PLY；与从原生 GaussianModel AST 提取的真实 writer 做逐字节对照，属性/顺序/header 不变。
- 记录真实 optimizer 调用、各参数组 Adam clocks、reset 步、VGG 步和权重身份，不用每步 loss.item() 增加同步。
- 默认不按旧0.5训练墙钟截断；confirmed 在优化器/VGG不足时拒绝出有效成绩。
- 完整 B 进程计时与完整 native 参考进程配对，TRAIN gate 计入B。关闭两端多余的训练后 TEST 快测；正式评测仍单独运行全部 TEST。
- ZIP 中缺少.git仍可运行，记录关键源文件 SHA256。修正文档中“入口尚未封装”、旧仓链接与 E2E 主门槛。

competition/upstream 保持原样。已检查的 densify 参数传播及 native reset 窗口原本正确，没有重复修改。未加入尚无组合证据的选图、尺度、EDGE、Faster-GS Adam 或球壳。

## 实验协议

新确认：D{3500,4000} × r{0,.15@1250}，统一5000步/VGG200/.1/.0008/interval100，bicycle、dl0b、dl3e，各seed2/3，共24新模型。不新跑 D1250。预算、输入、GPU UUID、并发和归档IO需匹配，不将共享卡耗时用于正式TIME结论。

最新 H3 独立复核：V150/.1总分38.81485 vs V200/.1 39.06484；fresh D4000 38.82398 vs fresh D3500 39.08661；V200权重.075/.125分别38.97863/38.93543，均低于既有.1对照。它们是研究代理分，不能当作新CLI完整B或最终13例成绩。

## 验证状态

- CPU：68项通过，包含原始50项及新增命令/计时/不完整训练拒绝/PLY逐字节兼容/官方结构与三组预览测试；上游快照校验包含在原始测试中。
- CUDA：H3 GPU2 / source407505c 的600步真实训练通过，实际 VGG401–600 共200次、foreach六组/FP32、.0008实际增密、native600有599调用；导出147,735,627字节PLY并由官方 renderer 重载，完整25张TEST及25张GT均输出。该600步检查缩短LR日程及VGG位置，只验证机制，不声称5K质量或最终TIME。见 records/delivery-runtime-gate-20261031。
- 完整 A→R→B 和新默认的统一13例/同机native30K完整B预算：尚未完成，不宣称全局最优或51.153已复现。

## 目录和后续版本

官方仓 master HEAD实查仍为 cb29814af7e260f03f7565e6edadb35bf42c5a93。顶层新增完整 baseline/、resplat/ 实体目录；网络、FastGS 和3dgs库与受保护快照逐字节一致，入口脚本负责连接已审计的自有实现。原始快照未动，Git对象自动共享相同文件内容。默认环境 profile 改为可移植 conda/resplat/fastgs，默认权重路径与官方 resplat/pretrained 对齐。

原目录后续13个未提交精简改动独立保存并push为 deepseek-final-working-snapshot / 7734515b9d7311d402fd1c677c1339cdb7879287。其r15、自动M1剪枝、scene-cache默认并非当前组合已验配方，未默认合入；历史代码及建议保留可审查，没有丢弃。用户已确认后续仅Codex修改仓库、依据实验决定最终setting。

新增官方 Faster-GS Adam组合验证：D3500/5K/VGG200/.1/r0 固定，其余保持一致，foreach vs官方组件；3 badcase×2seed=12 fresh，完整B及同范围native预算。旧C0/VGG100微小净收益不能代替该验证。正式SH网格顺延，Adam默认不提前切换。
