# 12 小时连续研究记录

用户授权：连续研究 12 小时，定期同步已完成实验结论，继续提出后续实验。研究 agent 只写本目录研究产物，读取仓库与远端实验记录；不实现算法、不启动 GPU 训练、不修改运行代码。

开始：**2026-10-30 05:43:57 +08:00**。计划结束：**2026-10-30 17:43:57 +08:00**。运行目标已建立，当前进行中；到期前不能把目标标为完成。

## 执行节奏

- 持续推进论文细读、项目适配与实验反证；约每 15–20 分钟检查新提交及完成记录，遇到重要新结果立即同步。
- 每 30–60 分钟形成一次有实质新增的结论/方案文件，并更新本记录；没有新实验时明确写“未发现新完成记录”，不把脚本/日志进度当完成证据。
- 所有实验状态以原始 JSON、输入/代码/GPU 身份和完整对照为准。研究可以异步继续，不等待 coding agent 完成才开始思考。
- 远端访问仅使用已有 SSH 身份，只读；不读取/输出凭据，不触碰合作者备用资源。

## 05:44 — 起始同步

本地 HEAD `0bc1ebd`；执行 `git pull --ff-only` 返回 Already up to date，保留原 README 改动及其他 agent 的未提交 FINAL-REPORT。

新进展：输入数据删除问题已由 coding agent 在 `be4537a` 修复，后续不再作为当前未修复项；新增 uint8 image cache / E0 A/B、初始化锐化 A/B、campaign T/U/MCMC 以及 eval13 的 baseline 对照。它们目前是代码与任务入口，尚需读取远端完成证据。

注意：`docs/FINAL-REPORT.md` 自述 B-only 已获用户确认，但 `ProblemDesc.md` 仍写主报告 E2E。研究记录暂保留两种范围及来源差异，不把另一 agent 的自述改写为本轮看到的用户指令；后续读取新的规则文件。

已确认主 GPU 的现有 SSH 可只读连通。正在定位其最新输出，不启动任何 GPU 作业。

当前研究优先级：先读新增实验结果；细读 SAD-GS 的真实结构控制实现/移植边界；再审查“锐化收益是否跨初始化和 seed 稳定”，避免继续把单个正结果当免费收益。

## 05:49 — 首批新结果与两个实质更新

已只读归档主机首批四条 eval13 候选 JSON 与两例截断图片错误，详见 SYNC_20261030_01；尚缺同场景 baseline，不形成泛化胜出结论。

coding agent 提交 `e773c9a` 改回 raw 初始化（自述61.1 vs60.2，配对 JSON 待核验）。本轮额外 CPU 静态审计发现归档锐化输入低于0.005的比例为 DL3DV31.83%、hw19.86%、counter41.59%，说明变换会大幅改变剪枝候选集合；这不是实际删除率，也不是质量因果证明。详见 OPACITY_AUDIT_20261030。

SAD-GS 深读已完成第一轮：补充专用统计 buffer、计时边界、标准 PLY 过滤属性问题，并将 E1 改成“诊断→一次限额干预→等子数对照”。详见 STRUCTURE_TRANSFER_20261030。继续研究，不结束12小时目标。

## 06:04 — 第二次实验同步与方案去重

新快照共7个eval13候选case完成（增加hw/billboard、basketball、car），仍没有完整baseline配对，两个DL3DV仍受截断图片影响；详见 SYNC_20261030_02。新评测节点文档采用“同卡类缓存预算”，已明确需要改为实际GPU UUID配对或仅作质量比较。

新增 OPTIMIZER_OPTIONS：将低成本单步信赖域作为E6，不照搬完整二阶框架；VIEW_SELECTION_FEASIBILITY：prepared work没有真实tracks，E2增加来源检查，并排除9份现有相机的主点偏移假设；FASTGS_MECHANISM_AUDIT：基线已有多视角误差结构控制，阈值5不是5视角票数，E4须明确差异，E7仅在支持归一化诊断成立后考虑。

研究持续进行，目标结束时间仍为17:43:57 +08:00；未达到12小时，不标完成。

## 06:07 — 新MCMC叙述与深度梯度前置问题

提交 `c3c1ee5` 在RESEARCH §15记录MCMC容量/预算网格：已试配置在约77秒达到与25–34秒FastGS接近或更差质量。暂维持FastGS优先；这是新增叙述证据，原始逐run JSON、GPU配对仍未同步，不能升级为“所有MCMC只在>250秒才有效”的普遍结论。

发现 `depth_reg.decode_depth_rgb` 仅detach归一化分母的梯度问题，CPU单高斯反例已保存：真实dD/da=0，当前代理为7.5（a=.4,z=3）。新增DEPTH_INTERFACE_AUDIT，要求先澄清expected/accumulated深度和修正梯度，再进行E3。源代码未修改。

后续工作：继续跟踪06:20左右的候选/baseline和raw/sharp原始证据；研究只拟外观/分阶段参数更新、预算内的几何可信度，避免新增loss堆叠；按现有约束持续至17:43:57。

## 06:16 — 新节点原始证据及E8

SYNC_20261030_03归档并核验新增节点55份JSON：campaignU两个场景各三组×三seed，前期结构对照稳定优于关闭/推迟；8组MCMC完整metrics/module_b支持暂不扩大同类网格。当前没有raw/sharp配对文件，缓存配对也尚缺；记录与docs自述的数值差异，不擅自归因为算分错。

新增SPARSE_OPTIMIZER：ICLR2026 AdamW-GS适配判断及E8，保留前1250步dense结构，先测optimizer占比再试后期可见性更新，明确FastGS现有sparse flag不是可直接切换的完整路径。新增CACHE_TIME_BOUNDARY：当前cache在B前独立解码/写盘，须测首次构建E2E；不能只凭B节省宣称+0.9是整条流水线净收益。

共享工作区新HEAD cf2a772，已读其62.01总结；当前质量固定时的65条件上界仍不能证明未来75不可达。继续执行12小时研究。

## 06:21 — 第8例与外观求解方案

SYNC_20261030_04归档H1第8例trashbin；仍缺候选/baseline配对，新增bicycle A失败未定位原因。新增节点读到campaignP16组旧锐化数据及billboard缓存一对；后者约65→63秒训练，与docs的25秒B记录不是同一run，不混拼归因。

新增APPEARANCE_SOLVER（E9）：固定几何和opacity后用逐高斯SH小系统做外观更新，与等时appearance Adam比较；先检查混合/clamp/显存/统计成本。CPU两高斯反例说明逐高斯投影不是全局整图精确最小二乘，不能把解析求解当无条件质量收益。

coding agent已新增campaignX五seed稳健性与E8 timebd入口，尚无完成证据。发现timebd的optimizer段包含结构操作，建议分离后再裁决E8；研究代码未改。下一次常规远端同步约06:40，期间继续论文/适配研究；目标仍进行中，结束时间17:43:57。

## 06:25 — 新P0：SH导出符号与真实pivot

新增SH_EXPORT_AUDIT：导出SH旋转的一阶z基函数与官方FastGS符号不同；1024方向CPU核查和独立不依赖numpy的标量反例均支持非平凡旋转不保持颜色。H1实际源码hash与所读固定源码相同，三个真实pivot的旋转算子差异均非零，counter较大。这里只证明接口问题，尚无真实场景最终质量增益。建议coding agent在wrapper进行F0–F2对照，受保护upstream原样保留；未修改运行代码。

DEPTH_INTERFACE补充alpha/匹配峰值/先验可靠性的区别，以及16个context sidecar在hw/counter全部训练视角中命中率不同；E3先修梯度、记录支持量。E8-L步频日程单列，不能把其结果当可见性稀疏Adam的裁决。研究继续。
