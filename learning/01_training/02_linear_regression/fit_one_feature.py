import torch


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """计算两个同形状、非空 Tensor 的均方误差。"""
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值与预测值的形状必须一致")

    if y_true.numel() == 0:
        raise ValueError("输入 Tensor 不能为空")

    return ((y_pred - y_true) ** 2).mean()


def main() -> None:
    # 准备数据：真实关系为 y = 2x + 1。
    inputs = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([3.0, 5.0, 7.0])

    # 参数只初始化一次。
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.01
    num_steps = 100
    log_interval = 10

    for step in range(num_steps):
        # 前向计算：预测并计算损失。
        predictions = w * inputs + b
        loss = mse(targets, predictions)

        # 反向传播：计算参数的梯度。
        loss.backward()

        # 更新参数，不记录更新操作的计算图。
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # 清除本轮梯度，为下一轮求导准备。
        w.grad.zero_()
        b.grad.zero_()

        # 用更新后的参数重新计算损失，保证日志中的数值相互对应。
        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                updated_predictions = w * inputs + b
                updated_loss = mse(targets, updated_predictions)

            print(
                f"第 {step + 1:3d} 次更新 | "
                f"w = {w.item():.6f} | "
                f"b = {b.item():.6f} | "
                f"MSE = {updated_loss.item():.6f}"
            )

    # 训练结束，使用学到的参数预测新输入。
    test_input = 4.0

    with torch.no_grad():
        test_prediction = w * test_input + b

    print(
        f"\n输入 {test_input:.1f} 的预测结果："
        f"{test_prediction.item():.4f}"
    )


if __name__ == "__main__":
    main()