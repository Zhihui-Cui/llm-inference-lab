"""M14·3｜给「我爱猫。」的一个验证窗口评分，观察评分前后的同一个参数。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder, batch_loss
from m03_text.m04_sequence_windows import validation_loader


def main():
    torch.manual_seed(42)
    model = TinyDecoder()
    model.eval()
    before = model.position_embedding.weight[0, 0].item()

    with torch.no_grad():
        for input_ids, target_ids in validation_loader:
            print("本批验证目标", target_ids.tolist())

            scores = model(input_ids)
            validation_loss = batch_loss(scores, target_ids)
            print("验证损失", validation_loss.item())

    after = model.position_embedding.weight[0, 0].item()
    print("同一位置参数，评分前后", before, after)


if __name__ == "__main__":
    main()
