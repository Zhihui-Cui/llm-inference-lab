# 文本、注意力与固定批次训练

**M13 固定小批次训练诊断已在指导下完成。** 综合作业的 80 步损失从 2.46381 降到 1.03427。[W11 实测与验收记录](../../reports/2026-11-20-W11-checkpoint.md) · [综合作业](assignments/m11_m13_assignment.py)。下一项为 M14 隔离文本训练与验证。

在 WSL 仓库根目录运行：

```bash
.venv/bin/python learning/02_text/assignments/m11_m13_assignment.py
```

## 从哪里找

| 学习问题 | 文件，按学习顺序 |
| --- | --- |
| 一个字如何成为编号？下一字的输入和答案怎么配？ | [M03 共享字符词表](char_vocab.py) → [M04 滑动窗口与错位标签](m03_text/m04_sequence_windows.py) |
| 一个位置能读哪些位置？Q、K、V 怎么训练？ | [M05–M06 单头注意力与因果掩码](m05_attention/m05_m06_causal_attention.py) → [M07 可训练的 QKV 与 batch](m05_attention/m07_trainable_qkv.py) |
| 多个头怎么拆分、合并？从文字输入怎么走到注意力？ | [M08–M09 多头与 SDPA 对照](m05_attention/m08_m09_multihead_attention.py) · [从字符编号接 QKV 的过渡练习](m05_attention/text_to_attention.py) → [M10 字符和位置向量](m05_attention/m10_token_position_embedding.py) |
| 怎么得到每个位置的表示，再用下一字打分？ | [M11 decoder block 与前缀检查](m11_decoder/m11_decoder_block.py) → [M12·1 “我爱 → 学”的单题损失](m11_decoder/m12_01_single_loss.py) → [M12·2 两句六题的平均损失](m11_decoder/m12_02_fixed_batch.py) |
| 同一批题反复训练，怎样看出训练链路接通？ | [M13 固定批次训练](m13_training/)；各个实验的打开顺序见下表。 |
| 学完整个阶段后，能否自己重写？ | [M11–M13 阶段作业与分题自测](assignments/m11_m13_assignment.py) |

## M13 · 一次打开一个文件

| 顺序 | 打开后做什么 | 本人进度 |
| --- | --- | --- |
| [1 · 观察位置表的梯度](m13_training/m13_01_position_grad.py) | 运行反向传播，对照用过的三行与其余五行。 | 已观测 0～2 行有梯度，3～7 行为零。 |
| [2 · 跟踪一次参数更新](m13_training/m13_02_sgd_step.py) | 用本次梯度预计位置表 [0,0] 的变化，再看实际差值。 | 已实测一次变化约 +0.00143512。 |
| [3 · 对同一批题重新评分](m13_training/m13_03_compare_loss.py) | 同一个模型更新一步，从相同 ID 再答一遍。 | 已实测同次运行 2.951108 → 2.827162。 |
| [4 · 手写短训练](m13_training/m13_04_short_train.py) | 复用同一个模型与优化器，每次更新后重新前向。 | 已手写并运行 30 步，3.600199 → 1.676202。 |

读第 3 课时，`inputs_batch` 是“我爱学 / 代码帮”的编号，`targets_batch` 是答案“爱学习 / 码帮助”的编号。`model` 是这次实验复用的四个部件，分别为字符表、位置表、decoder block 和词表预测层。`loss_batch` 表示更新前六道题的平均损失，`loss_after` 表示更新后同一批题的平均损失。每份脚本可单独启动，模型配置为 CPU、特征宽度 2、两头、FFN 宽度 4、dropout 为 0。

## 手写与查阅

每个新问题单开一个文件，先预测，写一小段，记录真实输出，再进入下一题。M13·4 做完后打开综合作业，在手写区逐题作答。题号可选 1 至 4，省略题号时会按顺序自测。完成后用自己的梯度、参数与损失解释固定批次拟合，对新句子的表现再用独立文本评价。

```bash
.venv/bin/python learning/02_text/assignments/m11_m13_assignment.py 1
```

需要回看基础时，可按具体问题查找。[训练与文本练习对照表](../previous_practice.md)

历史验收：[M03–M07](../../tests/verify_m03_m07.py) · [M08–M10](../../tests/verify_m08_m10.py) · [M11–M12](../../tests/verify_m11_m12.py)。M11–M12 复核继续使用原入口和相邻的共享词表。[decoder_block.py](decoder_block.py)

<details>
<summary>用以前笔记中的文件名查找新位置</summary>

| 原文件名 | 现在的位置 |
| --- | --- |
| `attention_from_text.py` | [m05_attention/text_to_attention.py](m05_attention/text_to_attention.py) |
| `attention_practice.py` | [m05_attention/m05_m06_causal_attention.py](m05_attention/m05_m06_causal_attention.py) |
| `m11_decoder.py` | [m11_decoder/m11_decoder_block.py](m11_decoder/m11_decoder_block.py) |
| `m11_m13_assignment.py` | [assignments/m11_m13_assignment.py](assignments/m11_m13_assignment.py) |
| `m12_fixed_batch.py` | [m11_decoder/m12_02_fixed_batch.py](m11_decoder/m12_02_fixed_batch.py) |
| `m12_single_loss.py` | [m11_decoder/m12_01_single_loss.py](m11_decoder/m12_01_single_loss.py) |
| `m13_01_position_grad.py` | [m13_training/m13_01_position_grad.py](m13_training/m13_01_position_grad.py) |
| `m13_02_sgd_step.py` | [m13_training/m13_02_sgd_step.py](m13_training/m13_02_sgd_step.py) |
| `m13_03_compare_loss.py` | [m13_training/m13_03_compare_loss.py](m13_training/m13_03_compare_loss.py) |
| `m13_04_short_train.py` | [m13_training/m13_04_short_train.py](m13_training/m13_04_short_train.py) |
| `multihead_attention.py` | [m05_attention/m08_m09_multihead_attention.py](m05_attention/m08_m09_multihead_attention.py) |
| `sequence_windows.py` | [m03_text/m04_sequence_windows.py](m03_text/m04_sequence_windows.py) |
| `token_position_embedding.py` | [m05_attention/m10_token_position_embedding.py](m05_attention/m10_token_position_embedding.py) |
| `trainable_attention.py` | [m05_attention/m07_trainable_qkv.py](m05_attention/m07_trainable_qkv.py) |

</details>
