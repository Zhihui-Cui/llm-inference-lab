import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算两个同形状、非空 Tensor 的均方误差。"""
    if predictions.shape != targets.shape:
        raise ValueError("预测值与真实值的形状必须一致")

    if predictions.numel() == 0:
        raise ValueError("输入 Tensor 不能为空")

    return ((predictions - targets) ** 2).mean()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """计算整个数据集的 MSE，不更新参数。

    本练习中，每个样本只有一个输出值。
    """
    model.eval()

    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            predictions = model(batch_inputs)
            batch_loss = mse(predictions, batch_targets)

            batch_size = batch_inputs.shape[0]

            # 先还原这一批的平方误差总和，再跨批次累计。
            total_squared_error += batch_loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("评估数据集不能为空")

    return total_squared_error / total_samples


def main() -> None:
    # 固定随机种子，方便在同一环境下重复观察实验结果。
    torch.manual_seed(42)

    # 1. 准备训练数据：真实关系为 y = x^2。
    # 在 [-2, 2] 内取 21 个点，间隔为 0.2。
    train_inputs = torch.linspace(-2.0, 2.0, 21).reshape(-1, 1)
    train_targets = train_inputs ** 2

    # 验证数据不参与参数更新。
    validation_inputs = torch.tensor([
        [-1.5],
        [-0.5],
        [ 0.5],
        [ 1.5],
    ])

    validation_targets = torch.tensor([
        [2.25],
        [0.25],
        [0.25],
        [2.25],
    ])

    # 2. 配对输入和真实值，并设置分批读取方式。
    train_dataset = TensorDataset(train_inputs, train_targets)
    validation_dataset = TensorDataset(validation_inputs, validation_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )

    # 训练集评估单独使用不打乱顺序的 loader。
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )

    # 3. 创建模型和优化器。
    mlp_model = nn.Sequential(
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

    learning_rate = 0.01
    num_epochs = 1000
    log_interval = 100

    optimizer = torch.optim.SGD(
        mlp_model.parameters(),
        lr=learning_rate,
    )

    # 4. 分批训练：每一批完成一次参数更新。
    update_count = 0

    for epoch in range(num_epochs):
        # evaluate() 会切到评估模式，因此每轮训练前切回来。
        mlp_model.train()

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()

            predictions = mlp_model(batch_inputs)
            loss = mse(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            update_count += 1

        # 5. 一轮结束后，用当前参数评估两个完整数据集。
        if (epoch + 1) % log_interval == 0:
            train_loss = evaluate(mlp_model, train_eval_loader)
            validation_loss = evaluate(mlp_model, validation_loader)

            print(
                f"第 {epoch + 1:3d} 个 epoch | "
                f"累计更新 {update_count:3d} 次 | "
                f"训练 MSE = {train_loss:.6e} | "
                f"验证 MSE = {validation_loss:.6e}"
            )

    # 6. 查看学到的参数。
    # print("\n最终权重：", mlp_model.weight.detach())
    # print("最终偏置：", mlp_model.bias.item())

    # 这是验证集中的一个样本，单独展示预测结果。
    check_inputs = torch.tensor([[1.5]])
    expected_value = 2.25

    mlp_model.eval()

    with torch.no_grad():
        predicted_value = mlp_model(check_inputs).item()

    absolute_error = abs(predicted_value - expected_value)

    print(f"\n输入 [1.5] 的预测值：{predicted_value:.6f}")
    print(f"真实值：{expected_value:.6f}")
    print(f"绝对误差：{absolute_error:.6e}")
    print("是否达到误差小于 0.05：", absolute_error < 0.05)


if __name__ == "__main__":
    main()