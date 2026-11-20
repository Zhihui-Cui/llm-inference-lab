"""M13·2｜位置 0 的第一个特征，走一步 SGD 后会变多少？
先用本次梯度预测变化量，再对照更新前后的同一个数。
"""

import torch

# 直接运行本课时，让 Python 也能找到 02_text 中前面学过的模块。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m11_decoder.m12_02_fixed_batch import TinyDecoder, fixed_examples, mean_token_loss


def main():
    inputs_batch, targets_batch = fixed_examples()
    model = TinyDecoder()
    # 这个列表选中同一模型的两张表、block 和词表预测层，沿用原实验的写法。
    params = [model.token_embedding.weight, model.position_embedding.weight,
              *model.block.parameters(), *model.lm_head.parameters()]
    optimizer = torch.optim.SGD(params, lr=0.1)
    loss_batch = mean_token_loss(model(inputs_batch), targets_batch)
    loss_batch.backward()

    before = model.position_embedding.weight[0, 0].item()
    grad = model.position_embedding.weight.grad[0, 0].item()
    optimizer.step()
    after = model.position_embedding.weight[0, 0].item()
    print("更新前", before, "梯度", grad, "更新后", after, "实际变化", after - before)


if __name__ == "__main__":
    main()
