"""M14·4｜用训练文本更新一步，比较同一个验证窗口在更新前后的损失。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder, batch_loss, make_optimizer, train_steps
from m03_text.m04_sequence_windows import train_loader, validation_loader


def score_window(model, input_ids, target_ids):
    """沿用上一课的两行计算，给传入的窗口评分。"""
    model.eval()
    with torch.no_grad():
        scores = model(input_ids)
        return batch_loss(scores, target_ids).item()


def main():
    torch.manual_seed(42)
    model = TinyDecoder()
    optimizer = make_optimizer(model)

    # 固定住“我爱猫。”的这个窗口，两次验证都使用它。
    validation_inputs, validation_targets = next(iter(validation_loader))
    before = score_window(model, validation_inputs, validation_targets)

    # 从 M04 的训练文本取一批。模型只在这一批上更新一次。
    train_inputs, train_targets = next(iter(train_loader))
    model.train()

    train_history = train_steps(model, optimizer, train_inputs, train_targets, steps=1)

    after = score_window(model, validation_inputs, validation_targets)
    print("同一训练批次的损失，更新前后", train_history)
    print("同一验证窗口的损失，更新前后", before, after)


if __name__ == "__main__":
    main()
