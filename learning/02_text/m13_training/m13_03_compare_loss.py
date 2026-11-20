"""M13·3｜同一个模型更新一步，再答原来的六道题，平均损失变了多少？
现在只看这个文件。先预测方向，运行后贴出末尾的两个数。
"""

import torch

# 直接运行本课时，让 Python 也能找到 02_text 中前面学过的模块。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m11_decoder.m12_02_fixed_batch import TinyDecoder, fixed_examples, mean_token_loss


def fixed_batch_loss(model, inputs_batch, targets_batch):
    # 从相同的输入 ID 开始。model(...) 每次都会重新查两张表、过 block、算词表分数。
    scores = model(inputs_batch)
    return mean_token_loss(scores, targets_batch)


def main():
    inputs_batch, targets_batch = fixed_examples()  # 输入是“我爱学 / 代码帮”，目标是“爱学习 / 码帮助”
    model = TinyDecoder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # M13·2 的四个模块全在 model 中

    loss_batch = fixed_batch_loss(model, inputs_batch, targets_batch)  # 第一次答六道题
    loss_batch.backward()
    optimizer.step()

    loss_after = fixed_batch_loss(model, inputs_batch, targets_batch)  # 同一个模型再答同样六道题
    print("更新前 loss", loss_batch.item(), "更新一步后 loss", loss_after.item())


if __name__ == "__main__":
    main()
