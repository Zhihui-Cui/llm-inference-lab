from pathlib import Path

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(42)

    # 1. 六个包裹，每个包裹只有重量这一个特征。
    # 输入形状：[6, 1]。
    inputs = torch.tensor([
        [1.0],
        [1.5],
        [3.0],
        [3.5],
        [5.0],
        [5.5],
    ])

    # 每个包裹的真实类别。
    # 0：轻件，1：中件，2：重件。
    # 标签形状：[6]，类型为整数。
    targets = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        dtype=torch.long,
    )

    # 2. 输入一个特征，输出三个类别分数。
    model = nn.Linear(1, 3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_steps = 2000
    log_interval = 200

    # 3. 训练循环。
    model.train()

    for step in range(num_steps):
        optimizer.zero_grad()

        # logits 形状：[6, 3]。
        logits = model(inputs)

        # 直接传入原始分数，不提前做 softmax 或 argmax。
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        if (step + 1) % log_interval == 0:
            print(
                f"第 {step + 1:4d} 次更新前 | "
                f"交叉熵损失 = {loss.item():.6f}"
            )

    # 4. 训练结束后，观察训练样本上的结果。
    model.eval()

    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

    correct_count = (predictions == targets).sum().item()
    accuracy = correct_count / targets.numel()

    class_names = ["轻件", "中件", "重件"]

    print("\n重量 | 真实类别 | 预测类别 | 轻件概率 | 中件概率 | 重件概率")

    for index in range(len(inputs)):
        true_class = targets[index].item()
        predicted_class = predictions[index].item()

        print(
            f"{inputs[index].item():4.1f} |   "
            f"{class_names[true_class]}   |   "
            f"{class_names[predicted_class]}   | "
            f"{probabilities[index, 0].item():7.1%}  | "
            f"{probabilities[index, 1].item():7.1%}  | "
            f"{probabilities[index, 2].item():7.1%}"
        )

    print(f"\n训练集准确率：{accuracy:.1%}")
    print("logits 形状：", logits.shape)
    print("targets 形状：", targets.shape)
    print("predictions 形状：", predictions.shape)

    # 新样本不参与训练，只观察模型的预测。
    check_inputs = torch.tensor([
        [1.8],
        [2.2],
        [3.8],
        [4.2],
    ])

    # 根据预先规定的重量规则给出真实类别。
    check_targets = torch.tensor(
        [0, 1, 1, 2],
        dtype=torch.long,
    )

    model.eval()

    with torch.no_grad():
        check_logits = model(check_inputs)
        check_probabilities = torch.softmax(check_logits, dim=1)
        check_predictions = check_logits.argmax(dim=1)

    print("\n新样本：")
    print("重量 | 真实类别 | 预测类别 | 轻件概率 | 中件概率 | 重件概率")

    for index in range(len(check_inputs)):
        true_class = check_targets[index].item()
        predicted_class = check_predictions[index].item()

        print(
            f"{inputs[index].item():4.1f} |   "
            f"{class_names[true_class]}   |   "
            f"{class_names[predicted_class]}   | "
            f"{probabilities[index, 0].item():7.1%}  | "
            f"{probabilities[index, 1].item():7.1%}  | "
            f"{probabilities[index, 2].item():7.1%}"
        )

    check_accuracy = (
        (check_predictions == check_targets).float().mean().item()
    )

    print(f"新样本准确率：{check_accuracy:.1%}")

    checkpoint_path = Path(__file__).resolve().parent / "artifacts" / "multiclass_checkpoint.pth"
    checkpoint_path.parent.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"\n保存位置：{checkpoint_path}")


if __name__ == "__main__":
    main()