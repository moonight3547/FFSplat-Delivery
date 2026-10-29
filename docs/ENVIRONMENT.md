# 环境与运行

Conda管理环境。Mac/Windows支持CPU预览、校验及SSH控制；CUDA训练/指标实测运行于Linux。不要将本机操作系统当成GPU执行环境。

## 控制端

```bash
conda create -n ffsplat-hackathon python=3.12 pip -y
conda activate ffsplat-hackathon
pip install numpy scipy pillow plyfile pytest torch
python -m pytest -q
python -m ffsplat_eval.run
```

## Linux/CUDA

`competition/profiles/organizer-fresh.json` 是可移植示例；`compshare-20261029.json` 仅记录现有服务器路径。复制profile适配Conda和设备，不改算法设置来迎合某台机器。

- ReSplat/MCMC：Python3.12、Torch2.7、gsplat1.5.3；ReSplat还需pointops、COLMAP、VGG-LPIPS等。
- FastGS：Torch2.4/CUDA12.4对应扩展，simple-knn、fused_ssim；具体约束由 `competition.environment` 生成。
- 赛方缺失FastGS CUDA扩展源码。当前使用外部 `https://github.com/fastgs/FastGS` 的 `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`；此前通过接口/GPU诊断，但不能证明与未提供的官方二进制完全相同。

先取得这一明确依赖（在本仓库根目录执行，third_party被忽略）：

```bash
git clone https://github.com/fastgs/FastGS.git third_party/FastGS
git -C third_party/FastGS checkout 44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f
git -C third_party/FastGS submodule update --init --recursive

# 仅规划，不安装；--execute需Linux和匹配CUDA工具链，详见--help
python -m competition.environment --runtime fastgs \
  --profile competition/profiles/organizer-fresh.json --work outputs/build \
  --fastgs-provider pinned-fastgs
python -m competition.environment --help
```

构建路径在outputs中，避免修改原始快照。环境验收分别在对应Conda环境执行：

```bash
python -m competition.doctor --runtime fastgs
python -m competition.doctor --runtime resplat
python -m competition.doctor --runtime mcmc
```

每条命令要在相应环境运行，不能在单一环境连续执行后就宣称全部就绪。doctor仅检查导入/接口/CUDA，首次新机器还需实际渲染/反向与指标验收。

原始上游shell脚本有作者机器的硬编码路径，使用本仓库包装入口。上游按字节快照冻结，运行前验证 `upstream-snapshot.json`，无需访问旧仓库或GitCode即可读取官方代码；不要直接修改其网络或文件。
