import torch

def mse(y_pre: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """计算两个同形状、非空 Tensor 的均方误差。"""
    if y_pre.shape != y_true.shape:
        raise ValueError("预测值与真实值的形状必须一致")

    if y_pre.numel() == 0:
        raise ValueError("输入 Tensor 不能为空")

    return ((y_pre - y_true) ** 2).mean()

def main():
    # 4 个样本，每个样本有 2 个特征。
    inputs = torch.tensor([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
    ])

    # 每个样本对应 1 个真实值，形状为 [4, 1]。
    targets = torch.tensor([
        [-4.0],
        [ 2.0],
        [ 0.0],
        [ 6.0],
    ])

    weights = torch.zeros((2, 1), requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)

    learning_rate = 0.05
    num_steps = 200
    log_interval = 20

    for step in range(num_steps) :
        predictions = inputs @ weights + bias
        loss = mse(predictions, targets)

        loss.backward()

        with torch.no_grad() :
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad

        weights.grad.zero_()
        bias.grad.zero_()

        if (step + 1) % log_interval == 0 :
            with torch.no_grad() :
                tmp_pre = inputs @ weights + bias
                tmp_loss = mse(tmp_pre, targets)

            print(
                f"第 {step + 1} 次更新 | "
                f"w = {weights.detach()} | "
                f"b = {bias.item()} | "
                f"MSE = {tmp_loss.item()}"
            )

    test_inputs = torch.tensor([[2.0, 1.0]])

    with torch.no_grad():
        test_predictions = test_inputs @ weights + bias

    predicted_value = test_predictions.item()
    expected_value = 8.0
    absolute_error = abs(predicted_value - expected_value)

    print(f"x_1 = 2, x_2 = 1 时，预测值：{predicted_value:.6f}")
    print(f"绝对误差：{absolute_error:.6e}")

    assert absolute_error < 0.05, "预测误差超过允许范围"

if __name__ == "__main__" :
    main()