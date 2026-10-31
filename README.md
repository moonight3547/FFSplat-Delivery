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

profile 格式参考 [compshare-20261029.json](competition/profiles/compshare-20261029.json)；其中路径属于历史机器，运行前应另建本机 profile。数据/环境准备见 [docs/ASSETS.md](docs/ASSETS.md)、[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。输出目录必须为空。ZIP 导出无需 `.git`，报告仍记录关键源文件 SHA256。

计时采用用户确认的工作口径：同 GPU、同场景、同预处理图像的 **native 随机初始化 FastGS 30K** 为预算，主时间分子为完整 B 进程（加载、初始化、训练、导出均计入），可选 TRAIN gate 也计入 B。A/R、评测、总墙钟另列。参考与候选都关闭训练后的重复 TEST 快测，统一在正式 render/evaluate 阶段评测；native 优化器、PLY、reset 窗口和 29,999 次 optimizer 调用保持不变。内部计时单列，不能混用两种范围。`--budget-seconds` 只用于同机同输入已经实测、且时间范围一致的预算。

默认完成全部 5000 步；显式 `--b-fraction` 提前截断的 confirmed 运行会失败，不能把少跑 VGG 的结果计为完整配方。PSNR/SSIM/VGG-LPIPS 使用官方代码和完整固定 TEST 集；选图、训练、停止均不使用 TEST RGB。代理分与官方最终分仍需区分。

当前参数有三 badcase、双种子的组合和权重验证依据；**新组合没有完成统一 13 场景确认**。旧配方的 clean13 均值 51.153 属于历史配置与内部计时，不是新默认或完整 B 计时的已测成绩。详见 [审查和实验协议](docs/DELIVERY-REVIEW-20261031.md)、[交接](docs/HANDOFF.md)；历史记录保留在 [DELIVERY.md](DELIVERY.md)。
