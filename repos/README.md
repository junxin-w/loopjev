# 第三方源码

训练 renderer 来自 [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev)，基准提交为 `ac91fbbf4f4a3d0eb6da0b2aa5fdffcf1a1af65c`。本地 Git 历史中的原始 `jev/api.py` 与训练准备记录中的 baseline SHA256 一致。

这里只保留训练所需的最小模块闭包：`jev/__init__.py`、`jev/api.py`、`jev/metrics.py`，以及 MIT 许可证。没有收录完整第三方仓库、训练框架、数据或模型。

其中 `__init__.py`、`metrics.py` 与该提交完全相同；`api.py` 是实际运行的 NOUL 双输入版本，其 SHA256 必须等于 `experiments/protocols/data.json` 的 `renderer_sha256`。

- 精确来源及字节哈希：`Open-Jev/PROVENANCE.json`
- 相对于上述固定提交的最小改动：`../patches/open-jev-noul-dual-v1.patch`
- 第三方原许可证：`Open-Jev/LICENSE`

运行时保留历史相对路径 `repos/Open-Jev`，因此原训练源码不需要改路径或修改来源指纹。
