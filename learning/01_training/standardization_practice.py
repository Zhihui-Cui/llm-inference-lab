from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算两个同形状、非空 Tensor 的均方误差。"""
    if predictions.shape != targets.shape:
        raise ValueError("预测值与真实值的形状必须一致")

    if predictions.numel() == 0:
        raise ValueError("输入不能为空")

    return ((predictions - targets) ** 2).mean()


def build_model() -> nn.Module:
    """每个样本一个输入特征，输出一个预测值。"""
    return nn.Linear(1, 1)


# 新增：只使用传入的统计量，不重新计算均值和标准差。
def standardize(
    inputs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """使用训练集的统计量标准化输入。"""
    return (inputs - mean) / std


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    """计算整个数据集的 MSE，不更新参数。

    本实验每个样本只有一个输出值。
    """
    model.eval()

    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            # 新增：进入模型前，使用训练集统计量标准化。
            scaled_inputs = standardize(batch_inputs, mean, std)
            predictions = model(scaled_inputs)

            # 标签保持原值。
            batch_loss = mse(predictions, batch_targets)

            batch_size = batch_inputs.shape[0]
            total_squared_error += batch_loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("评估数据集不能为空")

    return total_squared_error / total_samples


def main() -> None:
    torch.manual_seed(42)

    # 1. 准备原始数据。
    inputs = torch.linspace(-3.0, 3.0, 120).reshape(-1, 1)
    targets = 1.5 * inputs - 0.7

    dataset = TensorDataset(inputs, targets)

    # 2. 先划分数据，再计算训练集统计量。
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [80, 20, 20],
    )

    # 新增：indices 是训练样本在完整数据集中的编号。
    # 只取这些样本，不能使用完整 inputs 计算统计量。
    train_inputs = inputs[train_dataset.indices]

    input_mean = train_inputs.mean(dim=0, keepdim=True)

    input_variance = (
        (train_inputs - input_mean) ** 2
    ).mean(dim=0, keepdim=True)

    input_std = input_variance.sqrt()

    if (input_std == 0).any():
        raise ValueError(
            "训练集中存在标准差为 0 的特征，不能直接标准化"
        )

    print("训练集均值：", input_mean)
    print("训练集标准差：", input_std)

    # 3. 创建四个加载器。
    # 数据集里仍保存原始输入，进入模型前才标准化。
    batch_size = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(
        f"训练样本：{len(train_dataset)} | "
        f"验证样本：{len(validation_dataset)} | "
        f"测试样本：{len(test_dataset)}"
    )

    # 4. 创建模型和优化器。
    model = build_model()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.03,
    )

    num_epochs = 200
    log_interval = 20
    update_count = 0

    # 5. 训练。
    for epoch in range(num_epochs):
        model.train()

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()

            # 新增：输入标准化，标签不变。
            scaled_inputs = standardize(
                batch_inputs,
                input_mean,
                input_std,
            )

            predictions = model(scaled_inputs)
            loss = mse(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            update_count += 1

        if (epoch + 1) % log_interval == 0:
            # 训练集评估和验证集评估都使用同一组统计量。
            train_loss = evaluate(
                model,
                train_eval_loader,
                input_mean,
                input_std,
            )

            validation_loss = evaluate(
                model,
                validation_loader,
                input_mean,
                input_std,
            )

            print(
                f"epoch {epoch + 1:3d} | "
                f"累计更新 {update_count:4d} 次 | "
                f"训练 MSE = {train_loss:.6e} | "
                f"验证 MSE = {validation_loss:.6e}"
            )

    # 6. 新增：参数与预处理统计量一起保存。
    model_path = Path(__file__).with_name(
        "linear_standardized_checkpoint.pth"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
    }

    torch.save(checkpoint, model_path)
    print(f"\n保存位置：{model_path}")

    # 7. 创建新模型，并恢复参数和统计量。
    loaded_model = build_model()

    loaded_checkpoint = torch.load(
        model_path,
        weights_only=True,
        map_location="cpu",
    )

    loaded_model.load_state_dict(
        loaded_checkpoint["model_state_dict"]
    )

    loaded_mean = loaded_checkpoint["input_mean"]
    loaded_std = loaded_checkpoint["input_std"]

    print("恢复的均值：", loaded_mean)
    print("恢复的标准差：", loaded_std)

    # 8. 固定原始输入，检查保存前后预测是否一致。
    check_inputs = torch.tensor([
        [-2.4],
        [0.3],
        [2.1],
    ])

    model.eval()
    loaded_model.eval()

    with torch.no_grad():
        original_predictions = model(
            standardize(
                check_inputs,
                input_mean,
                input_std,
            )
        )

        # 使用文件恢复的统计量。
        loaded_predictions = loaded_model(
            standardize(
                check_inputs,
                loaded_mean,
                loaded_std,
            )
        )

    print("\n保存前的预测：")
    print(original_predictions)

    print("加载后的预测：")
    print(loaded_predictions)

    assert torch.allclose(
        original_predictions,
        loaded_predictions,
    ), "保存前后的预测不一致"

    print("通过：保存前后的预测一致")

    # 9. 用加载后的模型和统计量进行最终测试。
    total_squared_error = 0.0
    total_samples = 0
    max_absolute_error = 0.0

    loaded_model.eval()

    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            # 新增：测试输入也要使用保存的统计量。
            scaled_test_inputs = standardize(
                test_inputs,
                loaded_mean,
                loaded_std,
            )

            test_predictions = loaded_model(scaled_test_inputs)
            batch_loss = mse(test_predictions, test_targets)

            count = test_inputs.shape[0]
            total_squared_error += batch_loss.item() * count
            total_samples += count

            batch_max_error = (
                (test_predictions - test_targets)
                .abs()
                .max()
                .item()
            )

            max_absolute_error = max(
                max_absolute_error,
                batch_max_error,
            )

    if total_samples == 0:
        raise ValueError("测试数据集不能为空")

    test_mse = total_squared_error / total_samples

    validation_loss = evaluate(
        loaded_model,
        validation_loader,
        loaded_mean,
        loaded_std,
    )

    # 10. 查看标准化坐标中的参数。
    scaled_weight = loaded_model.weight.item()
    scaled_bias = loaded_model.bias.item()

    print(f"\n累计参数更新次数：{update_count}")
    print(f"标准化坐标下的权重：{scaled_weight:.6f}")
    print(f"标准化坐标下的偏置：{scaled_bias:.6f}")

    # 将参数换算回原始输入坐标，帮助理解。
    #
    # y = w * ((x - mean) / std) + b
    #   = (w / std) * x + (b - w * mean / std)
    #
    # 这里只计算用于展示的数值，不修改模型参数。
    original_weight = scaled_weight / loaded_std.item()
    original_bias = (
        scaled_bias - original_weight * loaded_mean.item()
    )

    print(f"原始输入坐标下的权重：{original_weight:.6f}")
    print(f"原始输入坐标下的偏置：{original_bias:.6f}")

    print(f"最终验证 MSE：{validation_loss:.6e}")
    print(f"最终测试 MSE：{test_mse:.6e}")
    print(f"测试集最大绝对误差：{max_absolute_error:.6e}")

    assert update_count == 2000, "参数更新次数不符合预期"
    assert validation_loss < 1e-4, "验证 MSE 未达标"
    assert test_mse < 1e-4, "测试 MSE 未达标"
    assert max_absolute_error < 0.05, "测试集最大绝对误差未达标"

    print("通过：本次数值检查全部达标")


if __name__ == "__main__":
    main()