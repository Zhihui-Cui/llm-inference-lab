"""M14·5｜给同一组窗口评分：分批计算和一次算完，平均 token 损失是否一致？"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder
from m03_text.m04_sequence_windows import make_loader, train_samples
from m14_validation.m14_04_one_train_step import score_window


def mean_token_loss(model, loader):
    """本课窗口等长、无 padding，每个目标位置都参与平均。"""
    total_loss = 0.0
    total_targets = 0
    for input_ids, target_ids in loader:
        batch_mean = score_window(model, input_ids, target_ids)
        target_count = target_ids.numel()  # 本批目标张量里有几个编号，就有几道题。
        total_targets += target_count
        total_loss += batch_mean * target_count
    return total_loss / total_targets


def main():
    torch.manual_seed(42)
    model = TinyDecoder()
    # 两次评的是同一组训练窗口，模型只评分。
    one_batch = make_loader(train_samples, batch_size=len(train_samples), shuffle=False)
    input_ids, target_ids = next(iter(one_batch))
    reference = score_window(model, input_ids, target_ids)
    print("一次评完", target_ids.numel(), "个目标的平均损失", reference)

    small_batches = make_loader(train_samples, batch_size=2, shuffle=False)
    result = mean_token_loss(model, small_batches)
    print("按小批次汇总的平均损失", result)
    print("两种算法的绝对差", abs(result - reference))


if __name__ == "__main__":
    main()
