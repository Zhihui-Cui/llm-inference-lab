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


def standardize(
    inputs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """使用给定的训练集统计量转换输入，不在这里重新计算统计量。"""
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
            # loader 保存原始输入；进入模型前才标准化。
            scaled_inputs = standardize(batch_inputs, mean, std)
            predictions = model(scaled_inputs)
            batch_loss = mse(predictions, batch_targets)

            batch_size = batch_inputs.shape[0]
            total_squared_error += batch_loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("评估数据集不能为空")

    return total_squared_error / total_samples


def main() -> None:
    # 固定随机种子，便于在同一环境下观察这次练习。
    torch.manual_seed(42)

    # 1. 准备数据：120 个样本，每个样本一个特征。
    # inputs 和 targets 的形状都是 [120, 1]。
    inputs = torch.linspace(-3.0, 3.0, 120).reshape(-1, 1)
    targets = 1.5 * inputs - 0.7

    dataset = TensorDataset(inputs, targets)

    # 2. 训练开始前，只划分一次。
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [80, 20, 20],
    )

    # 新增：只取训练集中的输入来计算统计量。
    # indices 是这份训练数据在完整 dataset 中的样本编号。
    train_inputs = inputs[train_dataset.indices]
    input_mean = train_inputs.mean(dim=0, keepdim=True)
    input_variance = (
        (train_inputs - input_mean) ** 2
    ).mean(dim=0, keepdim=True)
    input_std = input_variance.sqrt()

    if (input_std == 0).any():
        raise ValueError("训练集中存在标准差为 0 的特征，不能直接标准化")

    print("仅由训练集计算的均值：", input_mean)
    print("仅由训练集计算的标准差：", input_std)

    # 3. 创建四个加载器。
    # train_loader 和 train_eval_loader 使用同一份训练数据。
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

    # 4. 模型和优化器只创建一次。
    model = build_model()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.03,
    )

    num_epochs = 200
    log_interval = 20
    update_count = 0

    # 5. 固定训练 200 轮。
    for epoch in range(num_epochs):
        model.train()

        for batch_index, (batch_inputs, batch_targets) in enumerate(train_loader):
            optimizer.zero_grad()

            # 新增：只转换输入，batch_targets 保持原值。
            scaled_inputs = standardize(batch_inputs, input_mean, input_std)
            predictions = model(scaled_inputs)
            loss = mse(predictions, batch_targets)

            loss.backward()

            optimizer.step()

            update_count += 1

            # 每轮只观察第一批，避免输出太多。
            if epoch < 2 and batch_index == 0:
                print("更新后的权重（标准化坐标）：", model.weight.item())
                print("更新后的偏置（标准化坐标）：", model.bias.item())
                print("本批更新用的权重梯度：", model.weight.grad.item())
                print("本批更新用的偏置梯度：", model.bias.grad.item())

        # 内层循环结束后，每 20 轮检查并打印一次。
        if (epoch + 1) % log_interval == 0:
            train_loss = evaluate(
                model, train_eval_loader, input_mean, input_std
            )
            validation_loss = evaluate(
                model, validation_loader, input_mean, input_std
            )

            print(
                f"epoch {epoch + 1:3d} | "
                f"累计更新 {update_count:4d} 次 | "
                f"训练 MSE = {train_loss:.6e} | "
                f"验证 MSE = {validation_loss:.6e}"
            )

    # 6. 保存最后一轮参数，以及预测时必须复用的训练集统计量。
    # 使用新文件名，避免覆盖原来只保存 state_dict 的参数文件。
    model_path = Path(__file__).resolve().parent / "artifacts" / "train_standardized_regression.pth"
    model_path.parent.mkdir(exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_mean": input_mean,
        "input_std": input_std,
    }
    torch.save(checkpoint, model_path)

    print(f"\n参数保存位置：{model_path}")

    # 7. 创建相同结构的新模型，加载参数。
    loaded_model = build_model()

    loaded_checkpoint = torch.load(
        model_path,
        weights_only=True,
        map_location="cpu",
    )
    loaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    loaded_mean = loaded_checkpoint["input_mean"]
    loaded_std = loaded_checkpoint["input_std"]

    print("从文件恢复的均值：", loaded_mean)
    print("从文件恢复的标准差：", loaded_std)

    # 8. 对固定输入，检查保存前后的预测是否一致。
    check_inputs = torch.tensor([
        [-2.4],
        [0.3],
        [2.1],
    ])

    model.eval()
    loaded_model.eval()

    with torch.no_grad():
        original_predictions = model(
            standardize(check_inputs, input_mean, input_std)
        )
        # 重载后的预测使用文件恢复的统计量，不借用内存中的训练变量。
        loaded_predictions = loaded_model(
            standardize(check_inputs, loaded_mean, loaded_std)
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

    # 9. 使用加载后的模型进行最终测试。
    # 在一次遍历中累计平方误差，同时查找最大绝对误差。
    total_squared_error = 0.0
    total_samples = 0
    max_absolute_error = 0.0

    loaded_model.eval()

    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            scaled_test_inputs = standardize(
                test_inputs, loaded_mean, loaded_std
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

    # 使用重载后的模型和统计量，重新计算最终验证指标。
    validation_loss = evaluate(
        loaded_model, validation_loader, loaded_mean, loaded_std
    )

    # 10. 输出结果并检查数值标准。
    print(f"\n累计参数更新次数：{update_count}")
    print(f"标准化坐标下的权重：{loaded_model.weight.item():.6f}")
    print(f"标准化坐标下的偏置：{loaded_model.bias.item():.6f}")

    # y = w * ((x - mean) / std) + b
    #   = (w / std) * x + (b - w * mean / std)
    original_scale_weight = loaded_model.weight.item() / loaded_std.item()
    original_scale_bias = (
        loaded_model.bias.item()
        - original_scale_weight * loaded_mean.item()
    )
    print(f"换回原始输入坐标的权重：{original_scale_weight:.6f}")
    print(f"换回原始输入坐标的偏置：{original_scale_bias:.6f}")
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
