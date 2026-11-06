# 文本与 Attention 练习（M03–M10）

当前状态：2026-11-06 按指导下完成及代码验收通过收尾。独立程度、运行结果及限制见 [验收记录](../../reports/2026-11-06-W09-checkpoint.md)。

| 文件 | 内容 |
|---|---|
| [char_vocab.py](char_vocab.py) | M03：训练字符词表、编码/解码、未知字符 |
| [sequence_windows.py](sequence_windows.py) | M04：各划分内的错位窗口与 DataLoader |
| [attention_practice.py](attention_practice.py) | M05–M06：单头计算、因果正负对照、T=1；另含只训练输出层的演示 |
| [trainable_attention.py](trainable_attention.py) | M07：可训练 QKV、一次更新、batch 前向隔离 |
| [multihead_attention.py](multihead_attention.py) | M08–M09：拆头/合并/输出投影、SDPA 与短序列、batch 练习 |
| [attention_from_text.py](attention_from_text.py) | 字符 Embedding 接 QKV 的过渡演示；未加位置和因果 mask |
| [token_position_embedding.py](token_position_embedding.py) | M10：字符+绝对位置 Embedding 与因果 Attention，默认只替换未来字符 |

各脚本可从仓库根目录以 `python -B learning/02_text/<文件名>` 运行。完整长度、batch 参考对照、不同头数及倒序错误检查保存在 [M03–M07 验收](../../tests/verify_m03_m07.py)和 [M08–M10 验收](../../tests/verify_m08_m10.py)，避免把练习中打印出结果误认为所有边界已检查。

这些脚本是学习过程的归档；当前没有训练完整语言模型，输出向量也不是下一个字的 Embedding。
