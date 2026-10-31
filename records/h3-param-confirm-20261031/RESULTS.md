# H3参数确认30：保留D3500 + VGG200/.1

P1a 6个VGG150、P1b 12个fresh密度配对、P2 12个窄权重模型全部完成标准PLY、全官方TEST和独立终审。三case双seed的统一参数比较支持原D3500/VGG200/.1：150、D4000、.075/.125均未超过相应对照；默认未被实验脚本修改。

供Delivery纳入的精简证据是本报告、[verification-summary](verification-summary.json)、[宏分数JSON](macro-score-summary.json)和[30模型CSV](all30_metrics.csv)。[完整终审](independent-final-audit.json)绑定所有实际权重/步号、时钟、输入SHA和归档链；原始PLY/PNG继续保存在节点canonical，不需搬进Delivery。

## P1a：末150 VGG位于100和200之间

|设置|PSNR|SSIM|LPIPS|Q80|完整B内部秒|TIME20内部|内部工作总分|进程工作总分|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|V100/.1 original|24.5027|0.804367|0.204750|25.0015|44.536|13.5514|38.5528|38.1751|
|V150/.1 fresh|24.5147|0.804500|0.201700|25.6317|46.433|13.1831|38.8148|38.4280|
|V200/.1 original|24.5280|0.804900|0.199150|26.1844|47.981|12.8804|39.0648|38.7208|

V150相对V100：六配对Q80均正，宏Q+0.6303，B+1.8972秒，内部净分+0.2620。相对V200：六配对Q80均负，宏Q−0.5527，B省1.5477秒，内部净分−0.2500、进程口径−0.2928。保留200。

## P1b：D4000没有确认净增

|fresh设置|PSNR|SSIM|LPIPS|Q80|平均点数|完整B内部秒|内部工作总分|进程工作总分|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D3500/V200/.1 fresh|24.5370|0.805000|0.199100|26.2096|1,186,384|47.999|39.0866|38.7154|
|D4000/V200/.1 fresh|24.4840|0.804667|0.199383|26.0839|1,207,765|48.714|38.8240|38.4550|

D4000相对本线fresh D3500：ΔP−0.0530、ΔS−0.000333、ΔLPIPS+0.000283、宏ΔQ−0.1257，点数+21,381，完整B+0.7153秒；内部净分−0.2626，进程口径−0.2604，逐配对Q胜出2/6。没有用旧控制的微小I/O时差冒充收益，也没有逐case启用D4000。

## P2：窄权重仍未超过.1

|设置|PSNR|SSIM|LPIPS|Q80|完整B内部秒|内部工作总分|进程工作总分|
|---|---:|---:|---:|---:|---:|---:|---:|
|V200/W075 fresh|24.5445|0.805800|0.200067|26.0863|47.919|38.9786|38.6058|
|V200/.1 original|24.5280|0.804900|0.199150|26.1844|47.981|39.0648|38.7208|
|V200/W125 fresh|24.4913|0.803667|0.199000|26.0864|48.144|38.9354|38.5613|

相对原V200/.1，.075宏ΔQ−0.0981、内部净分−0.0862（Q配对3/6胜）；.125宏ΔQ−0.0980、内部净分−0.1294（2/6胜）。.125的LPIPS略降，但PSNR/SSIM与成本抵消收益。按协议以原.1为主对照，同时强制附上本线fresh D3500/.1复现诊断：它相对原.1宏Q+0.0252、B+0.0178秒；.075/.125相对该fresh对照的Q分别−0.1233/−0.1232，方向未反转。没有挑有利的控制或case。

## 固定身份和实际时钟

[预注册](../H3_PARAM_CONFIRM_20261031.md)固定原Uniform16、ReSplat/refine2/corrected-world初值和5000步TRAIN master；Big5k abs .0008/int100、r0、foreach、horizon5000、原SH16、fullres FP32不变。GPU0=DL3e、GPU1=DL0b、GPU3=bicycle，同case/seed始终同卡，seed反序。P1a只改VGG尾段为4851..5000；P1b固定200/.1只改density截止；P2固定D3500/VGG200只改.075/.125。不叠加ISP、相机采样、SH cadence或边框项。

P1b两臂全部fresh且相邻同GPU配对；P1a的100/200和P2的.1参考仅在完整输入/core/库/VGG权重/UUID/native预算身份核验后复用。[复用proof](reuse-proof.json)保留原测量边界。共同seed/master不保证完整CUDA前缀逐位相同，点数与相同配方重复的差异均保留。

D3500每个模型实际29次density（600..3400），五FG Adam state=4971、SH rest=305；D4000每个模型实际34次（600..3900），五FG=4966、SH=304。预期由固定native在optimizer前无条件换参及原SH16调用推导，再逐模型读取record核验。全部仍是5000 GS schedule和实际150/200次VGG，没有把时钟小于5000误当少训练，也没有把stock时钟硬套给D4000。

## 聚合、成本与证据边界

每setting先平均3case×2seed的原P/S/L再算Q80，随后加逐case同GPU native比值的TIME20均值。附casewise clip与逐配对Δ，不混用定义；本次两种Q数值一致。内部完整B配内部native、进程墙钟配进程native，全部30模型两种口径都低于预算。随机初始化FastGS30k native预算的同卡/同图/源/FP32身份沿用原完整proof；A/R预计算、TEST评价、归档和排队均不混入B。工作代理不是官方13case总分。

五个真实600步TRAIN gate（V150、D3500、D4000、.075、.125）通过实际VGG/weight/梯度有限性与clock审核，不导出PLY、不评TEST。600步未达到4000截止，不能单凭该gate宣称D4000正确；正式六个D4000的34事件与4966/304均实际验证。

全局终审b2df94f再次核对：30组合完整且唯一、所有真实sample权重/raw乘积/闭区间、5000主流逐步相同、标准PLY归档前有限及SHA、全官方TEST/PNG集合、三个进程exit0、六组clock、FP32/foreach/r0/无scheduled reset、R和prepared/cache前后SHA；保留小记录的完整file/dir集合和SHA与target全量复读proof绑定，原12控制PLY/record/metrics重新hash。

新30 canonical共2,760普通文件/10,874,840,043B（10.128GiB），H3源全复验/noFD后只清2,270份重复PLY/PNG，释放10,851,247,901B；全部小JSON、日志和恢复索引保留。

当前新30 canonical：wave01–08在N2 `/root/h3-param-confirm-20261031-archive-root/`，wave09–10在N2 `/root/gpufree-data/codex/archive/h3-param-confirm-20261031-v1/`。H3小记录根 `/root/codex-h3-param-confirm-20261031/outputs/confirm-v1/`，每波恢复索引在同outputs的`archive-evidence/waveNN/canonical.json`。旧/新档案全局共享原root10GiB+D5GiB，未另借15GiB，N2重I/O沿与Lite B相同的EX/SH锁互斥。

旧VGG36已按原proof保全、完整迁H2后再清N2重复大产物；原.1正式12仍留H3。旧H3两份600 smoke PLY与unused GPU3-DL0b native也在H2全核验后仅清副本，为D4000保守空间门槛腾出缓冲，未删唯一证据。迁移另立relocation-index，原proof没有重写。所有正式B窗口不做H3本地归档重I/O。

训练执行source dee6f24；接收/迁移helper b924292；独立终审b2df94f。控制器413669已complete，GPU2 SH门槛仅CPU待机并为Delivery600保留窗口，没有改变stock16默认。
