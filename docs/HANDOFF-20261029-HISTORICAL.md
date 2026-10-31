# 历史交接，仅作来源记录

以下是迁移前旧仓交接，不是当前任务状态。

# 新对话交接

新主仓库：**https://github.com/moonight3547/FFSplat-Hackathon（private）**。本地目录为本仓库根目录，后续从这里打开新对话，不再在Hackathon父目录混用两个历史仓库。

## 已完成与边界

1. 原始赛方代码完整快照保存于 `competition/upstream/`，HEAD `cb29814`；相对旧 `a1139a7` 只有 `baseline/README.md` 改变，训练代码完全相同。快照受逐文件SHA256保护，无需旧仓库的submodule即可读取。
2. 当前A/B适配器、Conda执行器、默认3例/可选13例配置与独立Eval合并在此仓库；旧mixed10、下载脚本、凭据、模型和数据不进入代码仓库。
3. 用户确认随机初始化FastGS30K为工作预算。包装器、MCMC、配对重用、Eval均检查算法/步数/初始化，使用 `reference_fastgs`；不修改赛方原始代码。
4. 三例新标定已全部完成，未启动新的策略搜索。目前没有本轮GPU训练任务需要继续等待。小型原始记录见 [fastgs-random30k-calibration.json](../records/fastgs-random30k-calibration.json)，经Eval研究分支 `c756c18` 回传。
5. 当前CPU校验50项通过；上游1835个文件逐字节校验通过。此干净仓库尚未在GPU上重新跑完整A/ReSplat/B，不能把CPU通过等同于新仓库CUDA端到端验收。

## 新实测预算与旧分数的更正

|场景|随机初始化FastGS30K（秒）|旧经典3DGS30K（秒，仅历史）|
|---|---:|---:|
|DL3DV 001dccbc…|102.68|547.06|
|hw/billboard|100.13|884.54|
|mipnerf360/counter|93.72|519.06|

标定使用赛方默认预处理分辨率及冻结train/test划分，初始化随机视锥约100K高斯，30,000真实更新、位置LR长度30K、densify_until_iter=15000。时间为FastGS `module_b.json.total_seconds`，包含脚本内加载/初始化/训练/导出；另保存进程墙钟。

当前GPU：`GPU-db080f3e-1f31-ec8d-821e-326757e8d1e1`。历史控制：`GPU-eff448d2-be95-ffeb-ee6a-59c22630a40d`。同主机名不等于同GPU。不能把新预算与旧A/B时间包装成同机实测分数。

[历史结果换分母诊断](../records/historical-rescore-diagnostic.json)仅为**跨GPU估算**：

|组别|PSNR / 分|SSIM / 分|VGG-LPIPS / 分|平均B秒|B-only TIME估分|B-only代理总估分|E2E代理总分|
|---|---|---|---|---:|---:|---:|---|
|O0|27.6383 / 10.1107|0.878867 / 14.3093|0.1728 / 15.4400|54.7733|9.8856|49.7456|超预算，不能给有效总分|
|A|27.6520 / 10.1216|0.879000 / 14.3200|0.172733 / 15.4533|54.0733|10.0495|49.9444|超预算，不能给有效总分|
|B|29.4970 / 11.5976|0.902367 / 16.1893|0.1239 / 25.2200|348.6084|超预算|无效估计|超预算|
|AB|29.5497 / 11.6397|0.901900 / 16.1520|0.1241 / 25.1800|338.0424|超预算|无效估计|超预算|

上述E2E超预算是把旧秒数代入新预算得到的诊断，尚非新GPU候选实测裁决。质量先均值后算分，TIME逐case算分再平均；不把三个预算平均后统一算TIME。复现诊断：`python -m scripts.rescore_historical`。**原55～63分总分不再作为当前口径成绩。**

## 下一轮建议（未启动）

- 先确认比赛最终TIME分子：最新赛方README强调E2E，旧题面写B-only；目前主报告E2E、B-only并列。用户确认的是FastGS参考算法及随机初始化，没有另外确认分子切换。不能把这些都写成官方已确认。
- 在当前GPU建立新O0完整测量及资产身份，保留102.68/100.13/93.72秒标定证据；若重新预处理导致几何/划分变化，预算也需重新标定。新仓库snapshot身份不同，禁止绕过旧控制的源码配对校验。
- 新预算只有约100秒。按E2E旧A/R已可能耗尽预算，先降低A（特别是hw的COLMAP/预处理）与加载导出耗时，比继续增加MCMC训练轮数优先。
- B先做短FastGS接续与时间受控MCMC，LR/CTF随真实可用时间调整。A的ranked16历史未改变选图，不能当成已有效策略；覆盖16图仍为备选。
- 不自动运行旧四组/13例大扫参，先把时间和baseline定义对齐，再制定最少实验。
- 外部文献思路的 A/B 分派与可用性判定见 [docs/PREPOST-AB-20261030.md](PREPOST-AB-20261030.md)：A 侧优先（选图覆盖/连通、hw 位姿诊断），B 侧只剩「初始化整理 + 固定拓扑短程 + LR 日程匹配」；学习型方法全部出局，且有 2 条引用 venue 未获证实。

## 远端资源（无凭据）

主要GPU：`ssh -p 22304 root@cpod-1wqzm4msan2q-s3.podtcp.compshare.cn`。认证使用用户本地凭据；仓库不保存密码。备用：`ssh -p 31563 root@183.147.142.40`，主要给合作者使用，有必要且确认空闲后才用。

当前主要GPU上的现有路径：
- 标定输出：`/workspace/organizer-assets/autoresearch-20261029/random-fastgs30k/`
- 标定日志：`/workspace/organizer-assets/autoresearch-20261029/random-fastgs30k.log`
- 运行本次标定的独立源代码：`/workspace/organizer-projects/FFSplat-FastGSReference`，Enhance `9ec2099`。
- 旧完整四组结果：`/workspace/organizer-assets/suite-v2/`，只作历史证据。
- 数据：`/workspace/organizer-assets/datasets`（默认3例）。权重：同级 `resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`。
- 本地数据/权重已全部在ModelScope发布，参见ASSETS；不再重复旧上传任务。

服务器Git曾通过控制端本地代理+SSH转发获取私有仓库；该代理不是持久服务，新对话需要重建正常Git认证通道。管理token留本机。新仓库尚未部署到该GPU，后续源代码通过GitHub拉取，不用SCP/stdin发送。

## 建议新对话开场

“请先读 AGENTS.md、ProblemDesc.md 和 docs/HANDOFF.md，核对FastGS随机初始化30K预算与TIME分子，继续设计在正确预算内超过baseline的最少实验。暂不占用合作者的备用GPU。”
