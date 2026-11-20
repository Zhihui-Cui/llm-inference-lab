from pathlib import Path

import torch
from torch import nn

from binary_metrics import classification_metrics


def main() -> None:
    torch.manual_seed(42)

    # 1. 准备训练数据。
    # 每行一个包裹，特征为重量；0 为普通件，1 为重件。
    train_inputs = torch.tensor([
        [1.0],
        [2.0],
        [4.0],
        [5.0],
    ])

    # BCEWithLogitsLoss 要求这里使用浮点数标签。
    train_targets = torch.tensor([
        [0.0],
        [0.0],
        [1.0],
        [1.0],
    ])

    # 2. 创建模型、损失函数和优化器。
    # 模型输出原始分数 logits，不直接输出概率。
    model = nn.Linear(1, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_steps = 1000
    log_interval = 100

    # 3. 训练：只有训练数据参与参数更新。
    model.train()

    for step in range(num_steps):
        optimizer.zero_grad()

        logits = model(train_inputs)
        loss = loss_fn(logits, train_targets)

        loss.backward()
        optimizer.step()

        # loss 来自本次参数更新之前的前向计算。
        if (step + 1) % log_interval == 0:
            print(
                f"第 {step + 1:4d} 次更新前 | "
                f"分类损失 = {loss.item():.6f}"
            )

    # 4. 准备验证数据，用于比较和选择阈值。
    validation_inputs = torch.tensor([
        [2.5],
        [2.9],
        [3.5],
    ])

    validation_targets = torch.tensor([
        [0],
        [0],
        [1],
    ])

    model.eval()

    with torch.no_grad():
        validation_logits = model(validation_inputs)
        validation_probabilities = torch.sigmoid(validation_logits)

    # 固定模型，只改变判断阈值。
    print("\n验证集：比较不同阈值")

    false_positive_cost = 1
    false_negative_cost = 5

    for threshold in [0.3, 0.5, 0.8, 0.9]:
        validation_classes = (
            validation_probabilities >= threshold
        ).long()

        metrics = classification_metrics(
            validation_classes,
            validation_targets,
        )

        total_cost = (
            metrics["fp"] * false_positive_cost
            + metrics["fn"] * false_negative_cost
        )

        print(
            f"阈值 {threshold:.1f} | "
            f"准确率 {metrics['accuracy']:.1%} | "
            f"误报 {metrics['fp']} | "
            f"漏报 {metrics['fn']} | "
            f"精确率 {metrics['precision']:.1%} | "
            f"召回率 {metrics['recall']:.1%} | "
            f"总代价 {total_cost}"
        )

    # 5. 搜索 0.00～1.00、间隔 0.01 的候选阈值。
    # 本次搜索的目标是验证集准确率最高。
    # 5. 根据验证集总代价选择阈值。
    false_positive_cost = 1
    false_negative_cost = 5

    best_cost = float("inf")
    best_thresholds = []

    for threshold_index in range(101):
        threshold = threshold_index / 100

        validation_classes = (
            validation_probabilities >= threshold
        ).long()

        metrics = classification_metrics(
            validation_classes,
            validation_targets,
        )

        total_cost = (
            metrics["fp"] * false_positive_cost
            + metrics["fn"] * false_negative_cost
        )

        if total_cost < best_cost:
            # 找到更低代价，重新记录最佳阈值。
            best_cost = total_cost
            best_thresholds = [threshold]

        elif total_cost == best_cost:
            # 代价相同，保留所有并列结果。
            best_thresholds.append(threshold)

    # 提前约定：并列时选择最小阈值。
    selected_threshold = min(best_thresholds)

    print("\n验证集阈值搜索结果")
    print(f"最低总代价：{best_cost}")
    print("并列最佳阈值：", best_thresholds)
    print(f"按并列规则选出的阈值：{selected_threshold:.2f}")

    print(f"本次固定使用的阈值：{selected_threshold}")

    # 6. 准备测试数据，评估固定模型和阈值的效果。
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

    model.eval()

    with torch.no_grad():
        test_logits = model(test_inputs)
        test_probabilities = torch.sigmoid(test_logits)
        test_classes = (
            test_probabilities >= selected_threshold
        ).long()

    # 复用同一个函数，统计全部分类指标。
    test_metrics = classification_metrics(
        test_classes,
        test_targets,
    )

    print(f"\n测试集评估，固定阈值：{selected_threshold}")
    print("重量 | 真实类别 | 重件概率 | 预测类别")

    for index in range(len(test_inputs)):
        print(
            f"{test_inputs[index].item():4.1f} | "
            f"{test_targets[index].item():8d} | "
            f"{test_probabilities[index].item():8.2%} | "
            f"{test_classes[index].item():8d}"
        )

    print(
        f"\nTP：{test_metrics['tp']} | "
        f"FP：{test_metrics['fp']} | "
        f"FN：{test_metrics['fn']} | "
        f"TN：{test_metrics['tn']}"
    )
    print(f"测试准确率：{test_metrics['accuracy']:.1%}")
    print(f"误报：{test_metrics['fp']} 次")
    print(f"漏报：{test_metrics['fn']} 次")
    print(f"测试精确率：{test_metrics['precision']:.1%}")
    print(f"测试召回率：{test_metrics['recall']:.1%}")

    # 7. 保存模型参数和固定阈值，供独立预测脚本使用。
    checkpoint_path = Path(__file__).resolve().parent / "artifacts" / "binary_classification_checkpoint.pth"
    checkpoint_path.parent.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "threshold": selected_threshold,
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"\n模型与阈值已保存到：{checkpoint_path}")


if __name__ == "__main__":
    main()
