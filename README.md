# LoopJev — NOUL dual-input Stage I

这是 `noul-dual-v1` 全量第一阶段训练的源码归档。训练入口、模型、编码器、损失及配置取自实际运行目录，按检查点中的 SHA256 逐文件核对；没有用旧版单输入代码替代，也没有重写训练器。当前版本不包含第二阶段早退门训练或各类基线的独立实现。

## 方法与配置

每条 NOUL 记录构造 `No`、`Yes` 两个独立候选输入。共享标量读出头在每轮循环后分别评分，以两个真实得分做 softmax；原始类别顺序仍为 `[no, yes]`。Choice、Score 的输入格式保持原样。详见 [输入及损失说明](docs/NOUL_PROTOCOL.md)。

| 项目 | 实际配置 |
|---|---|
| 底座 | 原始 Ouro-1.4B；从底座初始化，不加载旧第一阶段 adapter |
| 循环和监督 | 4 loop；每轮监督，按轮平均 |
| 训练数据 | 136,974 条：Choice 102,694 / NOUL 23,675 / Score 10,605 |
| 验证数据 | 25,413 条：Choice 11,798 / NOUL 11,133 / Score 2,482 |
| 数据遍历 | 1 epoch；有效 batch 16；共计划 8,561 次更新 |
| LoRA | `q_proj`、`v_proj`；rank 8 / alpha 16 / dropout 0 |
| 学习率 | LoRA 2e-4；读出头 1e-3；warmup 比例 0.03 |
| 损失 | CE + 0.1 × 按类别求和的 Brier；按记录及监督轮平均 |
| 输入长度 | 每个候选最多 2,048 token，超长拒绝，不截断 |
| 种子与数值条件 | seed 0；确定性开启；TF32 关闭 |

配置原件：[loop4 训练配置](experiments/configs/loopjev_noul_dual_v1_uniform_loop4_seed0.json)。源码归档完整性与模型训练完成是不同状态；本仓库不宣称本次训练已完成。导出时检查点进度见 `provenance/TRAINING_SOURCE_VERIFICATION.json`。

## 目录导航

| 路径 | 职责 |
|---|---|
| `experiments/run.py` | 原训练 CLI；本归档支持第一阶段 `preflight`、`train`、`evaluate` |
| `experiments/src/loopjev/models/decision.py` | Ouro 循环、共享标量读出、候选分组和 LoRA |
| `experiments/src/loopjev/data/records.py` | 固定成员、候选独立编码及长度约束 |
| `experiments/src/loopjev/training/tasks.py` | 候选 logits、CE/Brier、逐轮损失 |
| `experiments/src/loopjev/training/engine.py` | 全量训练及末尾完整验证 |
| `experiments/src/loopjev/training/checkpoints.py` | 检查点、恢复及版本核验 |
| `experiments/src/loopjev/input_protocol.py` | 防止旧输入协议、配置和产物混用 |
| `experiments/protocols/data.json` | 冻结数据、成员、tokenizer 和 renderer 哈希 |
| `experiments/manifests/main-v1/` | 固定成员清单，以 gzip 保存原始字节 |
| `repos/Open-Jev/` | 仅保留训练 renderer 所需的三个第三方模块及许可证 |
| `patches/` | NOUL 改动记录及最小 renderer 补丁 |
| `provenance/` | 来源、历史核验和本次归档完整性记录 |

CLI 原件还列有历史框架的其他子命令；它们不属于本次归档的支持范围。不要据此把本归档当作完整第二阶段或基线项目。

## 准备与运行

建议在 Linux 上使用原训练环境，或依据 [.envs/README.md](.envs/README.md) 重建环境。CPU 功能核验不替代目标 GPU 上的冒烟检查。

先恢复固定成员文件，并检查原始字节：

```bash
python tools/restore_manifests.py
python tools/verify_sources.py
python experiments/run.py train --help
```

然后把原始模型和数据绑定到仓库根目录下的相对路径，细节及固定版本要求见 [数据和模型准备](docs/SETUP.md)。不要修改冻结哈希以迁就不同数据或模型。

准备好环境、原始数据和模型后，先做输入预检：

```bash
python experiments/run.py preflight \
  --config experiments/configs/loopjev_noul_dual_v1_uniform_loop4_seed0.json \
  --data-lock experiments/protocols/data.json
```

全量训练命令如下。输出目录必须是全新的；执行前自行选择合适的 GPU。

```bash
python experiments/run.py train \
  --config experiments/configs/loopjev_noul_dual_v1_uniform_loop4_seed0.json \
  --data-lock experiments/protocols/data.json \
  --output experiments/outputs/training/loopjev_noul_dual_v1_uniform_loop4_seed0
```

仅在该输出目录已有本版本完整兼容的检查点时追加 `--resume`。模型、数据或源码不同会被恢复检查拒绝。整理及下载代码本身不会启动上述命令。

## 实验边界

模型权重、原始数据、token 缓存、训练输出和登录凭据不提交到仓库。仅归档可核对的配置、固定成员元数据和源码。全量训练不等于 NOUL-only 的 100 步 A/B 试验；本版本不收录后者的运行配置或将其结果作为完整训练结果。

旧单输入版本的门控、预测与温度文件不能当作本版本结果。新输入需重新提取第二阶段特征并训练、校准；正式 Test/OOD 不参与训练或调参。第三方模块的来源与改动见 [repos/README.md](repos/README.md)。
