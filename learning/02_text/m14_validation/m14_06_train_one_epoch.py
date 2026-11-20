"""M14·6｜训练文本练一整遍，“我爱猫。”的验证损失怎样变化？"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder, make_optimizer, train_steps
from char_vocab import train_text, validation_text
from m03_text.m04_sequence_windows import make_loader, train_samples, validation_loader
from m14_validation.m14_05_mean_token_loss import mean_token_loss


def print_losses(stage, model, train_for_scoring):
    # 两套题各自一次评完。mean_token_loss 只评分。
    train_loss = mean_token_loss(model, train_for_scoring)
    validation_loss = mean_token_loss(model, validation_loader)
    print(stage, "训练 45 题平均损失", train_loss, "验证 3 题平均损失", validation_loss)


def main():
    torch.manual_seed(42)
    model = TinyDecoder()
    optimizer = make_optimizer(model)
    print("用来训练的原文", train_text)
    print("只用来验证的原文", validation_text)

    train_for_scoring = make_loader(train_samples, batch_size=len(train_samples), shuffle=False)
    print_losses("训练前", model, train_for_scoring)

    # 按原文顺序训练：前七批各两个窗口，最后一批一个窗口。
    train_batches = make_loader(train_samples, batch_size=2, shuffle=False)
    model.train()
    for train_inputs, train_targets in train_batches:
        train_steps(model, optimizer, train_inputs, train_targets, 1)

    print_losses("训练一轮后", model, train_for_scoring)


if __name__ == "__main__":
    main()
