# FFSplat-Hackathon

干净的比赛开发入口：**赛方代码快照 + A/B 策略 + 默认三场景评测**。不依赖旧 Enhance/Eval 工作目录，也不包含旧 mixed10 实验框架。

先读 [当前题意](ProblemDesc.md)，再读 [新对话交接](docs/HANDOFF.md)。[来源与版本](references/SOURCES.json)可追溯到旧仓库，但旧结果不等于新口径成绩。

| 内容 | 入口 |
|---|---|
| 当前执行口径、约束和未确认项 | [ProblemDesc.md](ProblemDesc.md) |
| 赛方最新代码原样快照 | [competition/upstream](competition/upstream/README.md)，`cb29814` |
| 默认三场景与时间预算定义 | [benchmark.json](benchmark.json) |
| Conda / Linux CUDA / Mac、Windows 控制端 | [环境准备](docs/ENVIRONMENT.md) |
| 数据和权重 | [ModelScope 下载与校验](docs/ASSETS.md) |
| O0、A、B、AB 的运行入口 | `python -m ffsplat_eval.run` |
| 当前实验、限制、下一步 | [docs/HANDOFF.md](docs/HANDOFF.md) |
| 外部文献思路的 A/B 分派与可用性判定 | [docs/PREPOST-AB-20261030.md](docs/PREPOST-AB-20261030.md) |

## 当前状态

- 本仓库默认时间参考：**同 GPU、同场景、同预处理图像的随机初始化 FastGS 30,000 步**，不是 ReSplat 接续30K，也不是经典3DGS30K。这是用户于2026-10-29确认的工作口径。
- 赛方最新 README/`run_reference.sh` **仍写经典3DGS30K**。两者有差异，保留原文；本仓库包装入口落实用户口径，不把它冒称为官方已确认规则。
- 默认三例：DL3DV `001dccbc…`、hw/billboard、mipnerf360/counter；13例只提供可选配置，不宣称测过13例。
- 指标为 PSNR / SSIM / **VGG-LPIPS**。同时报告 A / ReSplat / B / E2E 秒数。当前主门槛为 E2E；B-only 单列诊断，百分制公式用于代理分，`official_score_100` 保持空。
- FastGS 随机30K已完成三例标定：102.68 / 100.13 / 93.72秒。旧 GPU UUID 已改变，历史质量指标可参考，旧总分和“全部有效”不能直接引用。精确状态见交接。

## 最短使用路径

```bash
# 查看默认协议，不启动训练
python -m ffsplat_eval.run

# 控制端 CPU 检查（Conda 环境）
python -m pytest -q

# Linux/CUDA上先登记本机资产；不要照抄旧机器的manifest
python -m competition.assets --spec benchmark.json \
  --data /assets/Hackathon-FFSplat-Enhance/datasets \
  --checkpoint /assets/Hackathon-FFSplat-Enhance/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth \
  --output outputs/assets.json

# 预览；确认路径、CUDA环境后，将describe改为run，使用新work目录
python -m ffsplat_eval.run --action describe \
  --data /assets/Hackathon-FFSplat-Enhance/datasets \
  --checkpoint /assets/Hackathon-FFSplat-Enhance/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth \
  --assets outputs/assets.json --profile competition/profiles/organizer-fresh.json \
  --work outputs/O0
```

`--policy mcmc-ctf` 为迁移B策略，`--a-policy ranked16` 为迁移A策略。默认O0是uniform16＋FastGS10K。`--action suite-plan` 仅预览四组；`--action suite` 才执行。初始状态配对必须来自当前代码、预算定义和GPU上的新控制组。

原题要求的最终 `run_pipeline.py --scene ... --output ...` 单场景提交接口尚待封装；当前可运行入口是上述benchmark CLI，不宣称已完成最终交付验收。
