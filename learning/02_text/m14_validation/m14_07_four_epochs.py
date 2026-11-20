"""M14·7｜从同一个起点连续训练四轮，每轮给同样的训练题和验证题评分。"""

from pathlib import Path
import sys
from time import perf_counter

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder, make_optimizer, train_steps
from char_vocab import train_text, validation_text
from m03_text.m04_sequence_windows import make_loader, train_samples
from m14_validation.m14_06_train_one_epoch import print_losses


def train_one_epoch(model, optimizer, train_batches):
    """上一课写过的循环：走完训练文本的 8 批，每批更新一次。"""
    model.train()
    for train_inputs, train_targets in train_batches:
        train_steps(model, optimizer, train_inputs, train_targets, steps=1)


def main():
    torch.manual_seed(42)
    model = TinyDecoder()
    optimizer = make_optimizer(model)
    train_batches = make_loader(train_samples, batch_size=2, shuffle=False)
    train_for_scoring = make_loader(train_samples, batch_size=len(train_samples), shuffle=False)
    print("训练原文", train_text)
    print("验证原文", validation_text)
    print_losses("第 0 轮（训练前）", model, train_for_scoring)

    start = perf_counter()
    for epoch in range(1, 5):
        train_one_epoch(model, optimizer, train_batches)
        print_losses(f"第 {epoch} 轮后", model, train_for_scoring)
    print("四轮训练和各轮评分用时（秒）", round(perf_counter() - start, 2))


if __name__ == "__main__":
    main()
