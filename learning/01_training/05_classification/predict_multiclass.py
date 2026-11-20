from pathlib import Path

import torch
from torch import nn


def main() -> None:
    checkpoint_path = Path(__file__).resolve().parent / "artifacts" / "multiclass_checkpoint.pth"

    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu",
    )

    # 类别名称的顺序必须与训练时一致。
    class_names = checkpoint["class_names"]

    # 使用与训练时相同的模型结构。
    model = nn.Linear(1, 3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inputs = torch.tensor([
        [1.8],
        [2.2],
        [3.8],
        [4.2],
    ])

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

    print("恢复的类别名称：", class_names)
    print("\n重量 | 预测编号 | 预测名称 | 轻件概率 | 中件概率 | 重件概率")

    for index in range(len(inputs)):
        class_id = predictions[index].item()

        print(
            f"{inputs[index].item():4.1f} | "
            f"{class_id:8d} | "
            f"{class_names[class_id]} | "
            f"{probabilities[index, 0].item():7.1%} | "
            f"{probabilities[index, 1].item():7.1%} | "
            f"{probabilities[index, 2].item():7.1%}"
        )


if __name__ == "__main__":
    main()