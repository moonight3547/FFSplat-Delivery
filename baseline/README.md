# FFSplat baseline

在 Linux NVIDIA CUDA 服务器运行，预先安装 Conda、Git、编译器和驱动。使用 Git LFS 克隆仓库取得原始权重；已有 checkout 可运行 `git lfs pull`。数据按官方目录放在 `../datasets/examples`，三组场景由 `scenes.json` 配置。

```bash
cd baseline
bash setup.sh        # 一次性：创建独立 ReSplat / FastGS 环境并编译扩展
bash run_train.sh    # 默认三组，A → ReSplat → B，保留训练记录
bash run_eval.sh     # 渲染、评测并生成 summary.json / summary.md / summary.csv
```

环境保存在 `../.runtime`，安装清单记录锁定版本、扩展和官方 Adam。可通过 `bash setup.sh --root /data/ffsplat-runtime` 指定环境存储位置；本仓自动记录其 profile。中断后使用相同参数加 `--resume` 续装。`bash setup.sh --plan` 仅预览。

产物位于 `baseline/results/<group>/<scene>/`，包含 `work/`、`results/{resplat,module_b}/` 和 `report.json`。训练、渲染和指标分别计时；评测不改写 B 的训练耗时或预算。

```bash
bash run_train.sh --describe                # 预览三组训练命令
bash run_train.sh dl3dv hw                  # 选择组
bash run_eval.sh dl3dv hw
bash run_train.sh --scene /data/mipnerf360/bicycle --output results/bicycle
bash run_eval.sh --output results/bicycle
```

默认 Uniform16、ReSplat refine2、FastGS 5K/D3500、VGG200/0.1、官方 Faster-GS Adam、均匀额外 L1=1。高斯尺寸保持原值。单场景入口默认训练后评测；三组入口将评测交给 `run_eval.sh`。官方原始代码和历史 README 保存在 `../competition/upstream`。
