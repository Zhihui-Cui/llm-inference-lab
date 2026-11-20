"""M12·单句｜读到“我爱”，怎样从词表分数算出预测“学”的损失？
复用 M11 返回的三行向量。运行后对照手算值、单位置 CE 和整句平均值。
"""

import torch
from torch import nn

# 直接运行本课时，让 Python 也能找到 02_text 中前面学过的模块。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id
from m11_decoder.m11_decoder_block import DecoderBlock, check_prefix


def inspect_single_loss(y_full, lm_head):
    # 同一个预测层逐行打分。每行两个特征 → 每个候选字一个原始分数。
    logits = lm_head(y_full)  # [3,11]；logits[i,j] 是位置 i 给编号 j 的分数
    print(logits)

    # 原文“我爱学习”的三个下一字依次是“爱、学、习”。targets[i] 指定第 i 行的正确列。
    # 列表推导按顺序查三个字的 ID；long 表示整数编号。targets 的形状是 [3]。
    targets = torch.tensor(
        [char_to_id[ch] for ch in "爱学习"],
        dtype=torch.long,
        device=logits.device,
    )

    # 单独检查位置 2。[1:2] 从下标 1 开始，取到下标 2 之前，保留“一行”的维度。
    # [1,11] 的分数配 [1] 的答案，得到这一个位置的损失。
    loss_fn = nn.CrossEntropyLoss()
    loss_pos2 = loss_fn(logits[1:2], targets[1:2])

    # 把刚才的计算展开来观察。[1] 取出位置 2 的 11 个分数，
    # dim=-1 沿候选字的维度计算概率；targets[1] 再选中“学”对应的那一列。
    probs_pos2 = torch.softmax(logits[1], dim=-1)
    p_xue = probs_pos2[targets[1]]
    print("读到“我爱”时，“学”的概率：", p_xue.item())
    # .item() 把单个数值取成 Python 数字，torch.log 是自然对数。
    print("用这个概率手算的损失：", -torch.log(p_xue).item())
    print("loss_pos2：", loss_pos2.item())

    # 交叉熵接收原始 logits。默认把三个位置的损失取平均，返回一个数。
    loss = loss_fn(logits, targets)
    # reduction="none" 保留三个损失。先创建这个损失函数，再用括号传入分数和答案。
    per_position_loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses_by_pos = per_position_loss_fn(logits, targets)
    print("三个位置各自的损失：", losses_by_pos.tolist())  # .tolist() 方便逐项阅读
    print("三个数的平均：", losses_by_pos.mean().item(), "整句 loss：", loss.item())


if __name__ == "__main__":
    block = DecoderBlock(d_model=2, d_ff=4, num_heads=2)
    y_full = check_prefix(block)
    lm_head = nn.Linear(2, len(char_to_id))
    inspect_single_loss(y_full, lm_head)
