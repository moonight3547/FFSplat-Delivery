# Hackathon Competition — ReSplat 加速重建 Baseline

比赛任务：在 ReSplat 前后接入模块 A / B，端到端（A → ReSplat → B）完成场景重建，
**总耗时需快于经典 3DGS 训练管线**。

## 额外数据获取地址

- [resplat 权重](https://drive.google.com/file/d/1HRZZWq-PyhhJ9zkDHhVL1IWdxnnIIjdY/view?usp=drive_link)
- [数据集](https://drive.google.com/drive/folders/1CZWTj9jHh9wVdUru0za69-Mtaz5Fzn16?usp=drive_link)

## 仓库结构

```
├── baseline/          # 官方 Baseline（详见 baseline/README.md）
│   ├── module_a/      #   A：位姿获取 + 均匀 16 视角选择
│   ├── run_resplat.py #   ReSplat：16 视角前馈重建 → 世界系 PLY
│   ├── module_b/      #   B：FastGS 接续训练 10000 步（全训练视角）
│   ├── fastgs/        #   vendored FastGS（fastgs conda env 使用）
│   ├── 3dgs/          #   vendored 经典 3DGS（30k 参考计时用）
│   ├── run_train.sh   #   一键端到端训练
│   ├── run_eval.sh    #   一键评测（PSNR/SSIM/LPIPS + 时间汇总）
│   └── run_reference.sh  # 可选：经典 3DGS 30k 从头训练参考
└── resplat/           # ReSplat 代码（含竖屏补丁，见 backup 说明）
```

## Baseline 成绩（RTX 4090 24GB）

| group | scene | PSNR | SSIM | LPIPS | 总耗时 | 经典30k参考 |
|---|---|---|---|---|---|---|
| dl3dv | 001dccbc… | 32.57 | 0.950 | 0.153 | **89s** | 547s / 29.91 |
| hw | billboard | 22.68 | 0.799 | 0.159 | **143s** | 915s / 16.75 |
| mipnerf360 | counter | 27.77 | 0.888 | 0.207 | **116s** | 531s / 27.08 |

## 快速开始

```bash
cd baseline
bash setup.sh        # 一次性：编译 3dgs CUDA 子模块（fastgs env）
bash run_train.sh    # 一键端到端训练
bash run_eval.sh     # 一键评测 + 汇总
```

环境、权重与数据集的下载说明见 `baseline/README.md`。

注意：
- ReSplat 权重（`resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`）需从主办方链接
  下载后放入 `resplat/pretrained/`（本仓库不含大文件）
- 数据集（`datasets/examples/{DL3DV,hw,mipnerf360}`）需从主办方链接下载后放到
  `baseline/../datasets/examples/`（与 `baseline/`、`resplat/` 平级）
