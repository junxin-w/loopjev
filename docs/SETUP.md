# 数据、模型与原始路径

本归档保留实际训练代码使用的相对路径与哈希。不要将冻结协议改为指向任意新文件；先准备完全相同的源文件，然后通过目录或符号链接提供相对路径。

## 原始模型

配置指向仓库根目录的 `models/Ouro-1.4B`。它应包含当时的原始非 Thinking 底座、模型远程代码与 tokenizer。模型文件和 tokenizer 的固定哈希分别见 `provenance/CHECKPOINT_MANIFEST.json` 与 `experiments/protocols/data.json`。

本条件独立初始化 LoRA 和共享读出头，不使用旧 LoopJev adapter 初始化。

## 原始数据

冻结数据协议中的 `data_sha256` 列出必需的原始文件相对路径与哈希，`dataset_revision` 固定当时的数据版本。`read_split` 读取 Parquet 的 `record_json` 列；仅有自行构造的同名数据文件不能通过完整性核验。

`experiments/manifests/main-v1/` 保存 Stage I、risk-train、risk-val、policy-fit 的固定成员元数据，不包含问题正文或训练标签。训练入口会核验所有成员清单，即使当前只进行第一阶段训练，也要恢复这四份文件。风险阶段的成员存在于冻结协议中，不代表本次执行了第二阶段。

成员清单以 gzip 形式纳入版本管理，执行：

```bash
python tools/restore_manifests.py
```

恢复工具按原始字节写回 `.json` 并核对协议中的 SHA256；不会覆盖内容不同的已有文件。不要手工重排 JSON、改缩进或换行后覆盖原件。

## 输出与恢复

`input_protocol.py` 要求输出在本仓库的 `experiments/outputs/` 之内，并在目录层级包含配置中的 `experiment_id`。同条件新启动必须使用未存在的输出目录；恢复使用原输出和 `--resume`。

本地源码不包含已有训练 checkpoint。需要恢复服务器上的训练时，继续使用服务器原 runtime，或者另行完整迁移相容 checkpoint、底座与数据。不要仅凭此源码归档宣称已迁移可恢复状态。

## 第三方 renderer

`repos/Open-Jev/jev` 只包含 `__init__.py`、`api.py`、`metrics.py` 三个训练所需模块。`api.py` 是实际 NOUL 双输入版本，文件 SHA256 与数据协议一致。没有提交完整 Open-Jev 仓库。原版本和补丁见 `repos/README.md`。
