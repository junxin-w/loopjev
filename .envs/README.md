# 训练环境

实际训练使用 Linux、Python 3.12 和 NVIDIA A800 80GB PCIe。核心包版本取自训练检查点保存的 `protocol.packages`，见 `requirements-training.txt`；这些是历史版本记录，不是经过跨平台安装验收的完整依赖锁。

优先复用原来的 `environments/loopjev/bin/python`。如另建环境，请先核对 CUDA/驱动与 PyTorch 安装，并安装核心包及其依赖，再进行 CPU 核验和独立 GPU 冒烟。不要为了安装方便改变实际训练源码或冻结数据协议。

模型依赖 Hugging Face 的本地 `trust_remote_code`；Ouro 模型目录必须保留原始 `configuration_ouro.py`、`modeling_ouro.py`、tokenizer 与权重，并核对来源和 SHA256。模型文件本身不包含在仓库中。

源码整理没有在本机安装或重建 GPU 环境，没有重新运行训练或完整 tokenizer 长度扫描。
