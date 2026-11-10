# NOUL 双候选输入与全量第一阶段训练

## 输入

旧版本每条 NOUL 记录只编码一个问题，得到标量 `s`，以 `[0, s]` 作为 `[no, yes]` 的 logits。这是合法的二分类参数化。

`noul-dual-v1` 改为两个独立输入，公共上下文与问题一致，候选分别是 `No` 与 `Yes`：

```text
Context:
{state}

Question: {question}
Proposed answer: {candidate}
Is this proposed answer correct? Answer Yes or No.
```

实际 `{state}`、`{question}` 的结构化渲染和 Ouro chat template 以源代码为准。候选之间不拼接：每个输入不包含另一个候选，也不包含目标标签。

原始记录的 `options` 必须仍为 `["no", "yes"]`；大写仅用于提示词显示。每轮共享读出头产生 `s_no`、`s_yes`，类别 logits 为 `[s_no, s_yes]`。最终概率由这两个得分共同归一化。

## 训练目标

对于记录 i、监督轮次 t，目标分布为 y，候选 softmax 概率为 p：

```text
L_it = -sum_k y_k log(p_k) + 0.1 * sum_k (p_k - y_k)^2
L = mean_records(mean_supervised_loops(L_it))
```

NOUL 仍按一条记录计权，不因两个候选输入变成两条独立的训练样本。Choice 和 Score 继续采用各自完整候选集合；读出头输出标量，因此不要求所有题目有相同的候选数量。

## 数据与核验范围

| 划分 | Choice | NOUL | Score | 总计 |
|---|---:|---:|---:|---:|
| 第一阶段训练 | 102,694 | 23,675 | 10,605 | 136,974 |
| 开发验证 | 11,798 | 11,133 | 2,482 | 25,413 |

`provenance/CPU_VERIFICATION.json` 记录此前完成的功能核验、真实 tokenizer 与缓存核验，以及改动 NOUL 提示词的完整长度扫描。该文件是历史证据，不表示本次重新运行了这些检查，也不表示完整训练已经完成。

## 比较时的边界

- 本版本同时改变了 NOUL 提示词语义和分类参照，不能把结果变化单独归因于移除固定零分数。
- 全量混合训练与 NOUL-only、100 步 A/B 试验是独立条件。
- 旧版单输入模型、缓存、检查点和早退门不能直接作为本版本的产物。
- 正式 Calibration 仅用于冻结方法后的温度拟合；Test/OOD 不参与训练、阈值选择或调参。
- 已保存的 CPU 核验不等于 GPU 训练与完整复现验收。本次已取回完整第一阶段源码、实际配置及检查点保存的核心依赖版本；额外归档检查见 `provenance/ARCHIVE_VERIFICATION.json`。本次没有重新执行 GPU 训练或完整 tokenizer 长度扫描。
