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
    """训练和加载时使用相同的模型结构。"""
    return nn.Linear(1, 1)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """评估整个数据集，不更新参数。

    本实验每个样本只有一个输出值，因此按样本数汇总 MSE。
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
    all_inputs = torch.linspace(-2.0, 2.0, 100).reshape(-1, 1)

    clean_targets = -2 * all_inputs + 3

    noise_std = 0.2
    noise = noise_std * torch.randn_like(clean_targets)

    all_targets = clean_targets + noise

    dataset = TensorDataset(all_inputs, all_targets)

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [60, 20, 20],
    )

    print(
        f"训练样本：{len(train_dataset)} | "
        f"验证样本：{len(validation_dataset)} | "
        f"测试样本：{len(test_dataset)}"
    )

    batch_size = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_model()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    num_epochs = 1000
    log_interval = 20
    patience = 20

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    update_count = 0

    best_path = Path(__file__).with_name("linear_noisy_weights.pth")

    for epoch in range(num_epochs):
        model.train()

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_inputs)
            loss = mse(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            update_count += 1

        # 所有批次训练结束后，检查验证成绩。
        validation_loss = evaluate(model, validation_loader)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        # 定期检查完整训练集，并打印日志。
        if (epoch + 1) % log_interval == 0:
            train_loss = evaluate(model, train_eval_loader)

            print(
                f"epoch {epoch + 1:4d} | "
                f"更新 {update_count:5d} 次 | "
                f"训练 MSE = {train_loss:.6e} | "
                f"验证 MSE = {validation_loss:.6e} | "
                f"连续未改善 {epochs_without_improvement} 轮"
            )

        # 达到等待上限，退出 epoch 循环。
        if epochs_without_improvement >= patience:
            print(
                f"\n在第 {epoch + 1} 轮提前停止："
                f"验证损失连续 {patience} 轮没有改善。"
            )
            break

    last_validation_loss = evaluate(model, validation_loader)

    best_model = build_model()
    saved_state = torch.load(
        best_path,
        weights_only=True,
        map_location="cpu",
    )
    best_model.load_state_dict(saved_state)

    loaded_validation_loss = evaluate(
        best_model,
        validation_loader,
    )

    print(f"\n最后一轮验证 MSE：{last_validation_loss:.8e}")
    print(f"最佳轮次：{best_epoch}")
    print(f"记录的最佳验证 MSE：{best_validation_loss:.8e}")
    print(f"重新加载后的验证 MSE：{loaded_validation_loss:.8e}")
    print(f"最佳参数保存位置：{best_path}")

    # 检查重载后能否复现记录的验证成绩。
    assert abs(
        loaded_validation_loss - best_validation_loss
    ) < 1e-7, "重新加载后的验证成绩与记录不一致"

    test_loss = evaluate(best_model, test_loader)
    print(f"\n最终测试 MSE：{test_loss:.8e}")

    print("最佳模型权重：", best_model.weight.detach())
    print("最佳模型偏置：", best_model.bias.detach())

    print("\n测试输入 | 真实值 | 最佳模型预测")

    best_model.eval()

    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_predictions = best_model(test_inputs)

            for index in range(test_inputs.shape[0]):
                print(
                    f"{test_inputs[index].item():8.3f} | "
                    f"{test_targets[index].item():8.4f} | "
                    f"{test_predictions[index].item():12.6f}"
                )


if __name__ == "__main__":
    main()