# Dedicated multi-GPU evaluation node

目标：把「改一版 module B → 拿到 13 例成绩」的循环压到 10 分钟以内。

## 为什么能这么快

模块 B 的评测**不需要** module A / ReSplat：它们每个场景只做一次，产物（prepared work 目录、
世界系 PLY、实测预算秒数）由准备机（H1）产出后随 manifest 下发。评测节点上每个场景只剩：

```
module B (~30 s)  +  render_test (~10 s)  +  official metrics (~15 s)  ≈ 55 s
```

13 场景 / 3 卡 ≈ 5 场景每卡 ≈ **5 分钟**（预算秒数已缓存，不需要重测参考）。

## Manifest

`records/evalnode-manifest.json` 是一个数组，每个场景四项：

| 字段 | 含义 |
|---|---|
| `case` | 场景 id，如 `hw/billboard` |
| `work` | module A 输出目录（含 `split.json`/`images`/`sparse`） |
| `init` | ReSplat 世界系 PLY |
| `budget_seconds` | 同卡类 FastGS 随机初始化 30k 秒数（预算，只测一次） |

新场景加入时：准备机上跑一次 A + ReSplat + 参考 30k，把 work/PLY 同步过来，并在 manifest 里加一行。

## 一轮评测

```bash
# 评测节点，repo 根目录
bash research/experiments/eval13_fast.sh <lane> <nlanes> records/evalnode-manifest.json outputs/round-<commit>
# 例：3 卡并行
for l in 0 1 2; do bash research/experiments/eval13_fast.sh $l 3 records/evalnode-manifest.json outputs/round-abc1234 & done; wait
```

输出每个场景的 B 秒数、PSNR/SSIM/VGG-LPIPS、质量分/TIME/总分，以及场景均值。
`RECIPE=...` `STEPS=...` `DENSIFY_UNTIL=...` 环境变量可覆盖配方，用于一版一分支的对照。

## 分支/提交约定

1. 每个改动一个分支：`git checkout -b exp/<topic>`，提交 info 里写清改了什么、期望什么。
2. 推送：`git push origin exp/<topic>`（远端已改为 SSH，之前 https 推送被 >100MB 的大文件挡住，
   已清理；`records/` 只存 JSON 证据，不存 PLY）。
3. 评测节点 `git fetch && git checkout <commit>` 后跑上面的命令，输出写 `outputs/round-<commit>`。
4. 只有 13 例均值提升且无场景回退 >1 分才合并。

## 当前覆盖

manifest 现有 3 个场景（三个默认例）。其余 10 个场景需要在准备机上依次跑 A + ReSplat
后才能加入；这是评测节点唯一的扩展成本。
