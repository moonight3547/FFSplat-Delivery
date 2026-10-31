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

- CPU：65项通过，包含原始50项及新增命令/计时/不完整训练拒绝/PLY逐字节兼容测试；上游快照校验包含在原始测试中。
- CUDA：待运行 H3 GPU2 的600步真实训练 smoke，检查实际200次VGG与优化器/精度/PLY审计；另以native短程核对 wrapper 不改变调用数。
- 完整 A→R→B 和新默认的统一13例/同机native30K完整B预算：尚未完成，不宣称全局最优或51.153已复现。
