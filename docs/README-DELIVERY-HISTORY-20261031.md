# 迁移前 README 历史记录

以下为旧方案和实验说明，当前入口以仓库首页与环境文档为准。

# FFSplat-Delivery

最终提交仓库（private）：https://github.com/moonight3547/FFSplat-Delivery 。原始源提交 `45feb68` 保存在 `deepseek-testing-45feb68`；实际收到的目录快照是 `d7a3754`。后续方案在此仓库更新。

当前默认：**Uniform16 → 原始 ReSplat（refine=2，修正 SH 导出基）→ BigD3500 / 5000 步 + 最后 200 步 VGG-LPIPS（0.1）**。关闭重分配与 opacity reset，FP32 / TF32 关闭，PyTorch foreach Adam，原生 SH 更新节奏和无损向量化 PLY 导出。完整参数见 [confirmed_recipe.json](configs/confirmed_recipe.json)。

```bash
# 无 CUDA、无数据也可预览真实 B 命令，不写输出
python run_pipeline.py --scene /data/mipnerf360/bicycle --output outputs/bicycle --describe

# 在 Linux CUDA 机器执行；profile 指定本机 conda 与环境，checkpoint 指定原始权重
python run_pipeline.py --scene /data/mipnerf360/bicycle --output outputs/bicycle \
  --profile competition/profiles/local-machine.json \
  --checkpoint /models/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth

# 显式复现旧 D1250 / 单次 r15 / 无 VGG 配方
python run_pipeline.py --scene /data/mipnerf360/bicycle --output outputs/legacy --preset legacy

python -m pytest -q
```

默认使用 [delivery.json](competition/profiles/delivery.json)：PATH 中的 conda、resplat/fastgs 两个环境；权重默认读取 resplat/pretrained，可用 --checkpoint 或 FFSPLAT_CHECKPOINT 覆盖。历史机器 profile 不作为默认。不同机器可另建本机 profile。数据/环境准备见 [docs/ASSETS.md](docs/ASSETS.md)、[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。输出目录必须为空。ZIP 导出无需 `.git`，报告仍记录关键源文件 SHA256。

计时采用用户确认的工作口径：同 GPU、同场景、同预处理图像的 **native 随机初始化 FastGS 30K** 为预算，主时间分子为完整 B 进程（加载、初始化、训练、导出均计入），可选 TRAIN gate 也计入 B。A/R、评测、总墙钟另列。参考与候选都关闭训练后的重复 TEST 快测，统一在正式 render/evaluate 阶段评测；native 优化器、PLY、reset 窗口和 29,999 次 optimizer 调用保持不变。内部计时单列，不能混用两种范围。`--budget-seconds` 只用于同机同输入已经实测、且时间范围一致的预算。

默认完成全部 5000 步；显式 `--b-fraction` 提前截断的 confirmed 运行会失败，不能把少跑 VGG 的结果计为完整配方。PSNR/SSIM/VGG-LPIPS 使用官方代码和完整固定 TEST 集；选图、训练、停止均不使用 TEST RGB。代理分与官方最终分仍需区分。

当前参数有三 badcase、双种子的组合和权重验证依据；**新组合没有完成统一 13 场景确认**。旧配方的 clean13 均值 51.153 属于历史配置与内部计时，不是新默认或完整 B 计时的已测成绩。详见 [审查和实验协议](docs/DELIVERY-REVIEW-20261031.md)、[交接](docs/HANDOFF.md)；历史记录保留在 [DELIVERY.md](DELIVERY.md)。

当前顶层按[官方 README](https://gitcode.com/gcw_yxxmv6uE/hackathon_competition)组织（2026-10-31 实查官方 HEAD `cb29814`）：

```text
baseline/
  module_a/                 # 官方预处理与 Uniform16
  run_resplat.py            # refine2 / SH 修正的提交入口
  module_b/                 # confirmed B 与 native 参考入口
  fastgs/                   # 原生 FastGS 源码
  3dgs/                     # 保留官方目录
  run_train.sh / run_eval.sh / run_reference.sh / setup.sh
resplat/                    # 原始 ReSplat 网络/配置
  pretrained/               # 外部原始权重，不纳入 Git
run_pipeline.py             # 单场景统一入口
configs/confirmed_recipe.json
competition/upstream/       # 原始只读快照，便于核对
```

`cd baseline && bash run_train.sh --describe` 可预览默认三组；单场景 `bash run_train.sh --scene ... --output ...` 使用相同默认。`bash run_eval.sh --output ...` 重做正式评测。仓库内保留研究历史；最终源码包通过 `python build_submission.py --output outputs/FFSplat-Delivery.zip` 从**已提交版本**导出，不含 Git、权重、数据及研究 raw/队列报告，附逐文件 SHA256 清单。ZIP 内文件与仓库同级，不额外嵌套一层目录。

官方 Faster-GS Adam 暂不默认加入：[既有六配对](records/official-adam-prior/RESULTS.md)只验证 C0/VGG100，完整 B 平均减少7.04%，Q80减少.27760、代理净增.25283；它不证明 BigD3500/VGG200 的组合收益。新的三场景双seed、12 fresh模型验证由实验 agent 跟进，通过后再由 Codex 裁决是否合入。
