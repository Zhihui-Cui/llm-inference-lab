import torch


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """计算两个同形状、非空 Tensor 的均方误差。"""
    if y_true.shape != y_pred.shape:
        raise ValueError("真实值与预测值的形状必须一致")

    if y_true.numel() == 0:
        raise ValueError("输入 Tensor 不能为空")

    return ((y_pred - y_true) ** 2).mean()


def main() -> None:
    # 真实关系：y = -3x + 2。
    inputs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    targets = torch.tensor([8.0, 5.0, 2.0, -1.0, -4.0])

    # 参数只初始化一次。
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.05
    num_steps = 200
    log_interval = 20

    for step in range(num_steps):
        # 前向计算：生成预测并计算损失。
        predictions = w * inputs + b
        loss = mse(targets, predictions)

        # 反向传播：计算损失对参数的梯度。
        loss.backward()

        # 根据梯度更新参数。
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # 清除本轮梯度，避免下一轮累加。
        w.grad.zero_()
        b.grad.zero_()

        # 使用更新后的参数计算日志指标。
        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                updated_predictions = w * inputs + b
                updated_loss = mse(targets, updated_predictions)

            print(
                f"第 {step + 1:3d} 次更新 | "
                f"w = {w.item():.6f} | "
                f"b = {b.item():.6f} | "
                f"MSE = {updated_loss.item():.6e}"
            )

    # 使用训练后的参数预测新输入。
    test_input = 3.0
    expected_output = -7.0
    tolerance = 0.05

    with torch.no_grad():
        test_prediction = w * test_input + b

    predicted_value = test_prediction.item()
    absolute_error = abs(predicted_value - expected_output)

    print(f"\n输入：{test_input:.1f}")
    print(f"预测结果：{predicted_value:.6f}")
    print(f"正确答案：{expected_output:.1f}")
    print(f"绝对误差：{absolute_error:.6e}")
    print(f"验收通过：{absolute_error < tolerance}")


if __name__ == "__main__":
    main()