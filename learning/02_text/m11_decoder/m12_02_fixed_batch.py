"""M12·两句｜六道预测题怎样共用同一个模型，并得到一个平均损失？
直接运行可核对“代 → 码”对应的分数行和目标。M13 会复用这里的固定 ID 和模型。
"""

import torch
from torch import nn

# 直接运行本课时，让 Python 也能找到 02_text 中前面学过的模块。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id, encode
from m11_decoder.m11_decoder_block import DecoderBlock


def fixed_examples():
    """返回两张 [2,3] 的整数表。第一张给模型读，第二张给预测打分。"""
    ids_batch = torch.tensor([
        encode("我爱学习", char_to_id),
        encode("代码帮助", char_to_id),
    ], dtype=torch.long)
    inputs_batch = ids_batch[:, :-1]   # 保留两句，去末字 → “我爱学”、“代码帮”
    targets_batch = ids_batch[:, 1:]   # 保留两句，去首字 → “爱学习”、“码帮助”
    return inputs_batch, targets_batch


class TinyDecoder(nn.Module):
    """把已经学过的四个模块放在一起。一次创建，以后反复用同一组参数。"""

    def __init__(self):
        super().__init__()
        self.block = DecoderBlock(d_model=2, d_ff=4, num_heads=2)
        self.lm_head = nn.Linear(2, len(char_to_id))
        self.token_embedding = nn.Embedding(len(char_to_id), 2)
        self.position_embedding = nn.Embedding(8, 2)

    def input_vectors(self, inputs_batch):
        position_ids = torch.arange(inputs_batch.shape[1], device=inputs_batch.device)
        token_vectors = self.token_embedding(inputs_batch)        # [2,3,2]
        position_vectors = self.position_embedding(position_ids)  # [3,2]，两句共用
        return token_vectors + position_vectors

    def forward(self, inputs_batch):
        x_batch = self.input_vectors(inputs_batch)
        hidden_batch = self.block(x_batch)
        return self.lm_head(hidden_batch)  # [2,3,11]，每个位置给 11 个候选字打分


def mean_token_loss(scores, targets_batch):
    scores_flat = scores.reshape(-1, scores.shape[-1])  # 一道题一行，保留候选字列
    targets_flat = targets_batch.reshape(-1)            # 相同顺序的六个正确字编号
    return nn.functional.cross_entropy(scores_flat, targets_flat)


def inspect_batch(logits_batch, targets_batch):
    logits_flat = logits_batch.reshape(-1, logits_batch.shape[-1])
    targets_flat = targets_batch.reshape(-1)
    # 展平后下标 3 是第四行，对应第二句的第一个位置，“代”预测“码”。
    torch.testing.assert_close(logits_flat[3], logits_batch[1, 0])
    assert targets_flat[3].item() == char_to_id["码"]
    print("分数和目标的形状：", logits_flat.shape, targets_flat.shape)
    return logits_flat, targets_flat


if __name__ == "__main__":
    inputs_batch, targets_batch = fixed_examples()
    model = TinyDecoder()
    logits_batch = model(inputs_batch)
    inspect_batch(logits_batch, targets_batch)
    print("六个位置的平均损失：", mean_token_loss(logits_batch, targets_batch).item())
