from pathlib import Path

import torch
from torch import nn


def main() -> None:
    checkpoint_path = Path(__file__).with_name(
        "binary_classification_checkpoint.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu",
    )

    # 创建与训练时相同的结构，再恢复训练好的参数。
    model = nn.Linear(1, 1)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 使用与模型一起保存的阈值。
    threshold = checkpoint["threshold"]

    inputs = torch.tensor([
        [2.8],
        [3.1],
        [4.5],
    ])

    model.eval()

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.sigmoid(logits)
        classes = (probabilities >= threshold).int()

    print(f"恢复的阈值：{threshold}")
    print("重量 | 重件概率 | 预测类别")

    for index in range(len(inputs)):
        print(
            f"{inputs[index].item():4.1f} | "
            f"{probabilities[index].item():8.2%} | "
            f"{classes[index].item():8d}"
        )


if __name__ == "__main__":
    main()