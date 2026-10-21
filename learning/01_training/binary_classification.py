import torch
from torch import nn
from pathlib import Path


def main() -> None:
    torch.manual_seed(42)

    # 每行一个样本：包裹重量。
    inputs = torch.tensor([[1.0], [2.0], [4.0], [5.0]])

    # 普通件为 0，重件为 1。这里使用浮点数标签。
    targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

    # 模型输出分数，仍然是 z = wx + b。
    model = nn.Linear(1, 1)

    # 新内容 1：分类损失，替换之前的 MSE。
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()

    for step in range(1000):
        optimizer.zero_grad()

        logits = model(inputs)

        # 直接传入原始分数。
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(
                f"第 {step + 1:4d} 次更新前 | "
                f"分类损失 = {loss.item():.6f}"
            )

    # 新输入：观察分界点附近的包裹。
    check_inputs = torch.tensor([
        [2.5],
        [2.9],
        [3.5],
    ])

    check_targets = torch.tensor([
        [0],
        [0],
        [1],
    ])

    model.eval()

    # 只计算一次概率，后面重复使用。
    with torch.no_grad():
        check_logits = model(check_inputs)
        check_probabilities = torch.sigmoid(check_logits)

    for threshold in [0.3, 0.5, 0.8, 0.9]:
        check_classes = (check_probabilities >= threshold).int()

        correct_count = (check_classes == check_targets).sum().item()
        accuracy = correct_count / check_targets.numel()

        # 预测为重件，但实际是普通件。
        false_positives = (
            (check_classes == 1) & (check_targets == 0)
        ).sum().item()

        # 预测为普通件，但实际是重件。
        false_negatives = (
            (check_classes == 0) & (check_targets == 1)
        ).sum().item()

        print(
            f"阈值 {threshold:.1f} | "
            f"准确率 {accuracy:.1%} | "
            f"误报 {false_positives} 次 | "
            f"漏报 {false_negatives} 次"
        )

        print(f"\n当前阈值：{threshold:.1f}")
        print("重量 | 重件概率 | 预测类别")

        for index in range(len(check_inputs)):
            print(
                f"{check_inputs[index].item():4.1f} | "
                f"{check_probabilities[index].item():8.2%} | "
                f"{check_classes[index].item():8d}"
            )

    best_correct_count = -1
    best_thresholds = []

    for threshold_index in range(101):
        threshold = threshold_index / 100

        # 根据当前阈值，把验证概率转换为类别。
        validation_classes = (
            check_probabilities >= threshold
        ).int()

        # 统计当前阈值下，有几个样本分类正确。
        correct_count = (
            validation_classes == check_targets
        ).sum().item()

        if correct_count > best_correct_count:
            # 成绩更好：更新最佳成绩，重新记录阈值。
            best_correct_count = correct_count
            best_thresholds = [threshold]

        elif correct_count == best_correct_count:
            # 成绩相同：追加一个并列最佳阈值。
            best_thresholds.append(threshold)

    best_accuracy = best_correct_count / check_targets.numel()

    print("\n验证集阈值搜索结果")
    print(f"最高准确率：{best_accuracy:.1%}")
    print(f"并列最佳阈值数量：{len(best_thresholds)}")
    print("并列最佳阈值：", best_thresholds)

    # 1. 固定此前根据验证集选出的阈值。
    selected_threshold = 0.8

    # 2. 准备新的测试样本和真实标签。
    test_inputs = torch.tensor([
        [1.5],
        [2.3],
        [2.8],
        [3.1],
        [3.4],
        [4.5],
    ])

    test_targets = torch.tensor([
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
    ])

    # 3. 用已经训练好的模型预测，不更新参数。
    model.eval()

    with torch.no_grad():
        test_logits = model(test_inputs)
        test_probabilities = torch.sigmoid(test_logits)
        test_classes = (
            test_probabilities >= selected_threshold
        ).int()

    # 4. 统计最终结果。
    correct_count = (test_classes == test_targets).sum().item()
    accuracy = correct_count / test_targets.numel()

    false_positives = (
        (test_classes == 1) & (test_targets == 0)
    ).sum().item()

    false_negatives = (
        (test_classes == 0) & (test_targets == 1)
    ).sum().item()

    print(f"\n测试集评估，固定阈值：{selected_threshold}")
    print("重量 | 真实类别 | 重件概率 | 预测类别")

    for index in range(len(test_inputs)):
        print(
            f"{test_inputs[index].item():4.1f} | "
            f"{test_targets[index].item():8d} | "
            f"{test_probabilities[index].item():8.2%} | "
            f"{test_classes[index].item():8d}"
        )

    print(f"测试准确率：{accuracy:.1%}")
    print(f"误报：{false_positives} 次")
    print(f"漏报：{false_negatives} 次")

    checkpoint_path = Path(__file__).with_name(
        "binary_classification_checkpoint.pth"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "threshold": selected_threshold,
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"\n模型与阈值已保存到：{checkpoint_path}")


if __name__ == "__main__":
    main()