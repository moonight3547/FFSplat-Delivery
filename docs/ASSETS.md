# 数据与权重

数据和权重均在 [ModelScope / MooNightXu/Hackathon-FFSplat-Enhance](https://modelscope.cn/datasets/MooNightXu/Hackathon-FFSplat-Enhance)。原 `datasets/` 树共3032文件、13场景；GitHub新仓库只存代码和小型记录。

```bash
git lfs install
git clone https://oauth2@www.modelscope.cn/datasets/MooNightXu/Hackathon-FFSplat-Enhance.git
cd Hackathon-FFSplat-Enhance
git lfs pull
```

如提示认证，在密码框输入自己的token；不把token写进URL、文档或Git配置。仓库内有 `DATASET_SHA256.json` 与 `SHA256SUMS`，必须取得LFS真实对象并校验，指针不算下载完成。

权重：`resplat-base-dl3dv-540x960-view16-a72dc6d0.pth`，896300265字节；SHA256：`a72dc6d0d56f7ccc50d145c17c83d866d4aaa2c13adb1a22f7a84481001e1a4c`。这是本地实测SHA，与文件名短前缀一致，不冒称赛方公布的完整SHA。

数据总字节数：5,693,070,006（原始目录文件总量）；ModelScope提供目录，不提供被中断的datasets.zip。`references/dataset-catalog.json`记录场景目录。按新 `benchmark.json` 重新生成 `competition.assets` 清单，旧清单带旧spec哈希，不直接复用。

原始赛方来源：
- 数据：https://drive.google.com/drive/folders/1CZWTj9jHh9wVdUru0za69-Mtaz5Fzn16
- 权重：https://drive.google.com/file/d/1HRZZWq-PyhhJ9zkDHhVL1IWdxnnIIjdY/view
