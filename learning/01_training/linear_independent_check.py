import json
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


def build_model() -> nn.Linear:
    """每个样本一个输入特征，输出一个预测值。"""
    return nn.Linear(1, 1)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """计算整个数据集的 MSE，不更新参数。

    本实验每个样本只有一个输出值。
    """
    model.eval()

    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            predictions = model(batch_inputs)
            batch_loss = mse(predictions, batch_targets)

            batch_size = batch_inputs.shape[0]
            total_squared_error += batch_loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("评估数据集不能为空")

    return total_squared_error / total_samples


def main() -> None:
    config = {
        "num_samples": 120,
        "input_min": -3.0,
        "input_max": 3.0,
        "target_slope": 1.5,
        "target_bias": -0.7,
        "split_sizes": [80, 20, 20],
        "batch_size": 8,
        "learning_rate": 0.03,
        "num_epochs": 200,
        "log_interval": 20,
        "seed": 42,
    }

    # 1. 固定本次实验的随机起点；划分还会单独保存实际索引。
    torch.manual_seed(config["seed"])

    inputs = torch.linspace(
        config["input_min"],
        config["input_max"],
        config["num_samples"],
    ).reshape(-1, 1)
    targets = config["target_slope"] * inputs + config["target_bias"]

    dataset = TensorDataset(inputs, targets)

    # 2. 训练开始前，只划分一次。
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        config["split_sizes"],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    # 3. 创建四个加载器。
    # train_loader 和 train_eval_loader 使用同一份训练数据。
    batch_size = config["batch_size"]

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
        lr=config["learning_rate"],
    )

    num_epochs = config["num_epochs"]
    log_interval = config["log_interval"]
    if num_epochs <= 0 or log_interval <= 0:
        raise ValueError("训练轮数和日志间隔必须为正整数")
    update_count = 0

    # 5. 按配置训练固定轮数；保存最后一轮，不做最优模型选择。
    for epoch in range(num_epochs):
        model.train()

        for batch_index, (batch_inputs, batch_targets) in enumerate(train_loader):
            optimizer.zero_grad()

            predictions = model(batch_inputs)
            loss = mse(predictions, batch_targets)

            loss.backward()

            optimizer.step()

            update_count += 1

            # 每轮只观察第一批，避免输出太多。
            if epoch < 2 and batch_index == 0:
                print("更新后权重：", model.weight.item())
                print("更新后偏置：", model.bias.item())
                print("权重梯度：", model.weight.grad.item())
                print("偏置梯度：", model.bias.grad.item())

        # 内层循环结束后，按配置的日志间隔检查并打印。
        if (epoch + 1) % log_interval == 0:
            train_loss = evaluate(model, train_eval_loader)
            validation_loss = evaluate(model, validation_loader)

            print(
                f"epoch {epoch + 1:3d} | "
                f"累计更新 {update_count:4d} 次 | "
                f"训练 MSE = {train_loss:.6e} | "
                f"验证 MSE = {validation_loss:.6e}"
            )

    # 单独计算最终验证结果，避免日志间隔导致结果遗漏或停留在较早轮次。
    validation_loss = evaluate(model, validation_loader)

    # 6. 保存本次实验的最后一轮参数、配置和实际划分。
    model_path = Path(__file__).with_name(
        "linear_independent_weights.pth"
    )
    torch.save(model.state_dict(), model_path)

    print(f"\n参数保存位置：{model_path}")

    split_indices = {
        "train": train_dataset.indices,
        "validation": validation_dataset.indices,
        "test": test_dataset.indices,
    }

    split_path = model_path.with_name("linear_split_indices.pth")
    torch.save(split_indices, split_path)

    config_path = model_path.with_name("linear_config.json")
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded_indices = torch.load(split_path, weights_only=True)
    for split_name, indices in split_indices.items():
        assert loaded_indices[split_name] == indices, f"{split_name} 索引读回不一致"

    loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert loaded_config == config, "配置读回不一致"
    print(f"配置保存位置：{config_path}")
    print(f"划分保存位置：{split_path}")
    print("通过：配置和三份划分索引保存、读取一致")

    # 7. 创建相同结构的新模型，加载参数。
    loaded_model = build_model()

    saved_state = torch.load(
        model_path,
        weights_only=True,
        map_location="cpu",
    )
    loaded_model.load_state_dict(saved_state)

    # 8. 对固定输入，检查保存前后的预测是否一致。
    check_inputs = torch.tensor([
        [-2.4],
        [0.3],
        [2.1],
    ])

    model.eval()
    loaded_model.eval()

    with torch.no_grad():
        original_predictions = model(check_inputs)
        loaded_predictions = loaded_model(check_inputs)

    print("\n保存前的预测：")
    print(original_predictions)

    print("加载后的预测：")
    print(loaded_predictions)

    assert torch.allclose(
        original_predictions,
        loaded_predictions,
        rtol=1e-5,
        atol=1e-8,
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
            test_predictions = loaded_model(test_inputs)
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

    # 10. 输出结果并检查数值标准。
    print(f"\n累计参数更新次数：{update_count}")
    print(f"最终权重：{loaded_model.weight.item():.6f}")
    print(f"最终偏置：{loaded_model.bias.item():.6f}")
    print(f"最终验证 MSE：{validation_loss:.6e}")
    print(f"最终测试 MSE：{test_mse:.6e}")
    print(f"测试集最大绝对误差：{max_absolute_error:.6e}")

    expected_updates = num_epochs * len(train_loader)
    assert update_count == expected_updates, "参数更新次数不符合预期"
    assert validation_loss < 1e-4, "验证 MSE 未达标"
    assert test_mse < 1e-4, "测试 MSE 未达标"
    assert max_absolute_error < 0.05, "测试集最大绝对误差未达标"

    print("通过：本次数值检查全部达标")


if __name__ == "__main__":
    main()