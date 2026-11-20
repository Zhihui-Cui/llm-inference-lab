"""M15·1｜给定前缀“我爱”，用最后位置的词表分数续写一个字。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder, make_optimizer
from char_vocab import char_to_id, encode, id_to_char
from m03_text.m04_sequence_windows import make_loader, train_samples
from m14_validation.m14_07_four_epochs import train_one_epoch


def prepare_model():
    """用 M14 相同的种子、训练窗口顺序和 32 次更新准备模型。"""
    torch.manual_seed(42)
    model = TinyDecoder()
    optimizer = make_optimizer(model)
    train_batches = make_loader(train_samples, batch_size=2, shuffle=False)
    for _ in range(4):
        train_one_epoch(model, optimizer, train_batches)
    model.eval()
    return model


def main():
    model = prepare_model()
    prefix = "我爱"
    input_ids = torch.tensor([encode(prefix, char_to_id)], dtype=torch.long)
    with torch.no_grad():
        scores = model(input_ids)

    print("给模型的前缀", prefix, "输入编号", input_ids.tolist())
    print("词表分数的形状", list(scores.shape))
    next_scores = scores[0, -1]
    next_id = next_scores.argmax().item()
    print("选中的词表编号", next_id, "续写结果", prefix + id_to_char[next_id])


if __name__ == "__main__":
    main()
