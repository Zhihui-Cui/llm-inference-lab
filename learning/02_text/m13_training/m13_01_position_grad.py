"""M13·1｜两句话各自预测，为什么位置表能收到两边的梯度？
先预测 8 行中哪些行参与查表。运行后对照前 3 行与后 5 行。
"""

# 直接运行本课时，让 Python 也能找到 02_text 中前面学过的模块。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m11_decoder.m12_02_fixed_batch import TinyDecoder, fixed_examples, mean_token_loss


def main():
    inputs_batch, targets_batch = fixed_examples()
    model = TinyDecoder()
    loss_batch = mean_token_loss(model(inputs_batch), targets_batch)

    loss_batch.backward()
    # 第 i 行对应位置编号 i，两列是该位置向量的两个特征。
    print("位置表 8 行 × 2 列的梯度", model.position_embedding.weight.grad, sep="\n")


if __name__ == "__main__":
    main()
