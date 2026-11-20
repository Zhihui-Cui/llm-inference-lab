"""M13·4｜连续更新同一套参数，固定六道题的平均损失会明显下降吗？
完成 M13·3 后再动手。复用那一课的 fixed_batch_loss，保留同一个模型和优化器。
"""

import torch

# 直接运行本课时，让 Python 也能找到 02_text 中前面学过的模块。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m11_decoder.m12_02_fixed_batch import TinyDecoder, fixed_examples
from m13_training.m13_03_compare_loss import fixed_batch_loss


def main():
    inputs_batch, targets_batch = fixed_examples()
    model = TinyDecoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_loss = fixed_batch_loss(model, inputs_batch, targets_batch)
    print("第 0 步，六道题的平均损失", initial_loss.item())

    # 你的练习区。先预测多步后的方向，再自己写短训练循环。
    # 每步清掉上一轮梯度，从固定 ID 重新算损失，反传后检查四个模块的梯度，再更新。
    # 最后一次更新也要重新前向。记录起点、中间、终点，并检查 loss 和梯度均有限。
    # 解释同一 batch 上的下降能说明什么。用新文本评估留到 M14。
    loss = initial_loss
    for step in range(1, 31):
        optimizer.zero_grad()
        # 1. 从手头的 loss 反向传播
        loss.backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} 没有收到梯度"
            assert torch.isfinite(parameter.grad).all(), f"{name} 的梯度里出现了 NaN 或无穷"

        # 2. 让优化器更新一次参数
        optimizer.step()
        # 3. 用同一个模型和固定 ID 重新评分，把新损失存回 loss
        loss = fixed_batch_loss(model, inputs_batch, targets_batch)
        print("第", step, "步，六道题的平均损失", loss.item())


if __name__ == "__main__":
    main()
