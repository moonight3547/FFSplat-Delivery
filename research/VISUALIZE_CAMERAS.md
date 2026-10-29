# 相机轨迹可视化

现成 13-case 网页：[`advisory/visualizations/cameras-20261030/index.html`](advisory/visualizations/cameras-20261030/index.html)。可直接在本地浏览器打开，无须启动服务器。13 张单场景俯视 PNG 和总览先生成，网页随后生成。Plotly JS 随目录保存在 `assets/`，离线使用时保留整个目录。

```bash
python -m pip install numpy matplotlib plotly
python -m research.visualize_cameras \
  --snapshot research/advisory/evidence/20261030-camera-snapshot.json \
  --out research/advisory/visualizations/cameras-20261030

# 单个已准备的场景；不会修改 work 或读取图片
python -m research.visualize_cameras --work outputs/my-case/work \
  --case my-case --lpips 0.2993 --out /tmp/my-camera-report

python -m unittest research.test_visualize_cameras
```

本轮临时依赖环境：`/private/tmp/ffsplat-camera-viz/bin/python`。若默认 Matplotlib 缓存目录不可写，可设 `MPLCONFIGDIR=/tmp/ffsplat-viz-mpl`；禁用 Python 缓存可用 `python -B`。

网页只挂载一个 3D iframe，左侧切换场景。右侧可切换记录的 context、uniform、FPS 和 coverage；图例可以打开全部光轴或绿色最近两中心邻接边。默认显示 train/test、文件名顺序连线、context 编号、视锥和紫色 pivot。pivot 为 `context[8]`，编号从 0 开始；少于 9 张时使用中间视角。

红色表示 **LPIPS 得分为 0**：`40 × clip((0.25 − LPIPS)/0.20, 0, 1)`。这与实际 LPIPS=0 不同。缺失指标显示未知，不按零分处理。页面中的 LPIPS 来自历史异条件运行，不能作为同条件 13-case 排行榜，也不参与任何选图计算。

本轮四场景保留冻结 split，另九场景从真实位姿重建 `llffhold=8` 划分及默认 uniform16 context；后者明确标为“重建划分 / 默认 context，非原实验 context 凭证”。真实位姿来源和哈希、指标来源、context 列表都能在页面下方展开，或从 `camera_diagnostics.json` 检查。

俯视平面由相机平均 up 方向 `-R_c2w[:,1]` 估计；平均方向一致性弱于 0.2 时使用相机中心 PCA 最小方差轴。它不是实测重力，线状轨迹的 PCA 平面可能不唯一。3D 使用同一等尺度坐标系，保留真实角度和相对距离。相机采用 COLMAP/OpenCV 的 +Z 朝前、+Y 朝下约定；视锥远平面距离仅是显示尺寸。

文件名 / split 顺序不是已验证的采集时间，灰线不代表网络邻接。默认隐藏的绿色边连接每个 context 的最近两个中心，对应当前网络 `local_mv_match=2` 的几何候选关系；它不证明特征匹配成功。coverage 是包围相机中心的视锥代理，不含深度、遮挡或 SfM 点轨迹。

量化诊断报告最大 train 到 context 中心距离、最小 context 间距、光轴覆盖空缺、相邻文件顺序光轴转角和索引空缺。距离按 train 非零两两距离的中位数归一化；位置与朝向各自匹配最近 context。它们用于发现聚集、空缺和转角，不能单独归因 LPIPS。只有 train 位姿进入三种对比策略，test 位姿仅显示；不读取 test RGB，也不启动训练。
