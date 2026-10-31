# BigD3500 + VGG200：建议采纳官方 Adam，保留显式 foreach 回退

12个fresh模型（3 badcase × 2 seeds × 2 backends）与独立审计全部通过。按用户确认的完整B工作代理，官方Adam净增 **0.55995**，六个配对全部为正；完整B均值减少 **3.25790秒（6.68%）**。建议将官方组件作为这个固定组合的显式优化器选择，并保留foreach可复现入口。它不是质量无损替换：宏Q80减少0.048，DL0b两seed质量均降；本结果没有证明统一13场景总分。

## 主口径

每个backend先平均3case×2seed原PSNR/SSIM/VGG-LPIPS，再算Q80；TIME逐case用匹配fresh native完整B分母后平均。两臂全部在预算内，official_score_100=null。

|backend|PSNR|SSIM|LPIPS|Q80|完整B秒|TIME20|完整B代理总分|
|---|---:|---:|---:|---:|---:|---:|---:|
|foreach|24.545333|.8052167|.1988833|26.276933|48.744030|13.089220|39.366154|
|official|24.541167|.8051167|.1990667|26.228933|45.486130|13.697169|39.926102|
|official−foreach|−.004167|−.0001000|+.0001833|−.048000|−3.257900|+.607949|**+.559949**|

内部B单列为45.065333→41.741333秒，内部TIME13.550092→14.187901，内部代理39.827025→40.416834（+.589809）；不把内部和进程秒数混算。casewise clipping诊断与本数据宏Q80数值相同，定义仍分别保存。最终高斯均值1,191,150→1,188,263（−2,887）；固定结构规则不意味着最终点集逐位相同。

## 六个配对（official−foreach）

|case|seed|ΔPSNR|ΔSSIM|ΔLPIPS|ΔQ80|Δ完整B秒|Δ完整B代理|
|---|---:|---:|---:|---:|---:|---:|---:|
|dl3e|2|-0.032|-0.0002|-0.0002|-0.0016|-3.4590|+0.6723|
|dl3e|3|+0.038|+0.0004|-0.0003|+0.1224|-2.8021|+0.6683|
|dl0b|2|+0.000|-0.0007|+0.0011|-0.2760|-3.3034|+0.3747|
|dl0b|3|-0.014|-0.0001|+0.0008|-0.1792|-2.8936|+0.3908|
|bicycle|2|-0.010|+0.0005|-0.0008|+0.1920|-3.9311|+0.8614|
|bicycle|3|-0.007|-0.0005|+0.0005|-0.1456|-3.1581|+0.3922|

仅2/6质量分更好、3/6 LPIPS更好；六对完整B与净代理均改善，最小净增+.37469。两个seed不足以给稳定置信区间；不能按case挑backend或把微小质量差称为逐位等价。全部原始指标见 [12行CSV](all12_metrics.csv)，完整逐配对及两计时范围见 [macro-score-summary.json](macro-score-summary.json)。

## 身份与实际验证

基于Delivery407505c；GPU0=DL3e / GPU1=DL0b / GPU3=bicycle。全部固定原Uniform16、refine2、修正SH world R、原context/reference、5000同TRAIN master。BigD3500、abs.0008/int100、horizon5000/r0/reset5001、末4801..5000的VGG200/.1/fullres FP32、matmul和cudnn TF32均关闭，原SH16与无损vecPLY。没有叠新采样、cache、ISP、EDGE或尺度。

两臂均经独立wrapper选择后端；原random-pop继续，主render前唯一AST插入应用TRAIN master，移除该插入后全部原AST相同，density scorer/RNG不变。Mac Python3.9独审补齐GPU Python3.12的空type_params字段后，原/修改两AST SHA均精确匹配；没有忽略不匹配字段。seed2 foreach→official，seed3 official→foreach；冻结共同seed/相机流不保证CUDA所有状态逐位一致。

官方pinned d23bb784d8f19b2432d847b80418d56f0494f599，仅原FusedAdam类和CUDA Adam kernel，不换rasterizer/ADC/PPISP或backward融合。actual class SHA9ea0b472…、CUDA SHA f7fed10a…、actual so SHA11e85983…均重读核验。复用的隔离二进制源于原N3 Torch2.4cu124/ABI0/CPython3.12、SM86构建；H3 SM89实际import、64步非零/小/零/None梯度和append/prune moment+clock迁移重验通过，数值不是逐位等价。

每个正式模型实际29次density，五FG state4971、SH state305、SH调用312；六个official模型均实际调用25,160次官方kernel，foreach均只触发multi-tensor dispatch。末200次VGG与实际weight.1、VGG参数冻结/RNG恢复/权重SHA、所有最终PLY属性finite、896张完整官方PNG均独立复读通过。初始R/prepared/cache像素与SHA前后不变。

fresh native random30K也在同三卡并发标定，原PIL/optimizer/PLY、29999调度和3000/6000/9000/12000 resets：DL3e完整114.060695/内部110.92秒；DL0b112.819458/109.71；bicycle130.498130/127.44。实际UUID、输入、线程8/8、precision双off、并发时间窗已审计；不借其它GPU或旧实验分母。[独立终审](independent-final-audit.json)、[输入前验](preflight.json)、[预算](fresh-native-budgets.json)。

首次wave1评测壳漏建render-model目录，三次render退出1；修复后只重评，不重训，原B wall保持。训练源17d00f0，评测恢复bdf087c；随后proof的SCP半写竞态由fdce8e8临时名→原子rename修复，从wave2续跑，不覆盖旧失败记录/重放模型。三formal训练源码字节相同，所有失败和恢复日志保留。

## 生产接线与存储

`research/adam_backend.py`提供显式AdamBackend('foreach'/'official',manifest)上下文、原training_setup factory hook及最终实际class/module/.so SHA、官方source、dispatch/kernel calls、六组clocks审计。requested official缺manifest/不匹配/导入失败会报错；显式foreach是回退，不能静默冒充official。仅SHA工具导入转为既有competition.contracts，优化器方法保持GPU验证版本；主run_pipeline/train_confirmed没有修改。

`research/install_official_adam.py`只Git fetch固定官方SHA，编译至独立target（MAX_JOBS≤4）或验证复用二进制，不pip升级共享env，不带管理凭据。真实脚本安装v2及GPU2 helper600两臂已通过：VGG200，FG599/SH37、官方3032kernel、显式foreach实际dispatch；[helper-smoke-proof.json](helper-smoke-proof.json)。新target manifest在H3 `/root/gpufree-data/codex/official-adam-production-helper-20261031-v2/manifest.json`。主线接线由root完成；默认训练长度仍5K。

1086普通文件/4,350,828,135B正式raw已四波直接归Mac `FFSplat-Delivery-OfficialAdam/private/archives/h3-official-adam-composite-20261031`，完整文件/目录集合+SHA/bytes目标复读与H3源复读/noFD通过后，仅清重复PLY/PNG，所有小证据与恢复索引保留。三份native的25文件/335,735,792B在H3 D保留，并有Mac完整复读副本。原VGG.1十二模型及Delivery600证据未动；没有去重/硬链接。每FS保≥2GiB，Mac cap10GiB/留100GiB；[存储证明](storage-proof-summary.json)。H3四卡已quiet交root做历史cache工程门槛，正式SH网格继续顺延。
