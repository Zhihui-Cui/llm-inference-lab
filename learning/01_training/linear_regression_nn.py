import torch
from torch import nn
from pathlib import Path

def mse(y_pre: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """计算两个同形状、非空 Tensor 的均方误差。"""
    if y_pre.shape != y_true.shape:
        raise ValueError("预测值与真实值的形状必须一致")

    if y_pre.numel() == 0:
        raise ValueError("输入 Tensor 不能为空")

    return ((y_pre - y_true) ** 2).mean()

def main() -> None:
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

    validation_inputs = torch.tensor([
        [ 2.0,  1.0],
        [-2.0,  0.0],
        [ 0.0,  3.0],
    ])

    validation_targets = torch.tensor([
        [ 8.0],
        [-3.0],
        [10.0],
    ])

    model = nn.Linear(2, 1)
    learning_rate = 0.05

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    num_steps = 200
    log_interval = 20

    for step in range(num_steps):
        optimizer.zero_grad()

        predictions = model(inputs)
        loss = mse(predictions, targets)

        loss.backward()

        optimizer.step()

        if (step + 1) % log_interval == 0:
            with torch.no_grad():
                updated_predictions = model(inputs)
                train_loss = mse(updated_predictions, targets)

                # 用模型预测 validation_inputs。
                val_predictions = model(validation_inputs)
                
                # 与 validation_targets 计算 val_loss。
                val_loss = mse(val_predictions, validation_targets)

            print(
                f"第 {step + 1:3d} 次更新 | "
                f"训练 MSE = {train_loss.item():.6e} | "
                f"验证 MSE = {val_loss.item():.6e}"
            )

    test_inputs = torch.tensor([[2.0, 1.0]])

    with torch.no_grad():
        test_predictions = model(test_inputs)

    predicted_value = test_predictions.item()
    expected_value = 8.0
    absolute_error = abs(predicted_value - expected_value)

    print(f"x_1 = 2, x_2 = 1 时，预测值：{predicted_value:.6f}")
    print(f"绝对误差：{absolute_error:.6e}")

    assert absolute_error < 0.05, "预测误差超过允许范围"

    # 参数文件放在当前脚本旁边，扩展名为 .pth。
    model_path = Path(__file__).with_suffix(".pth")

    print("准备保存的参数：", model.state_dict())
    torch.save(model.state_dict(), model_path)

    print("保存位置：", model_path)

    loaded_model = nn.Linear(2, 1)

    saved_state = torch.load(
        model_path,
        weights_only=True,
        map_location="cpu",
    )
    loaded_model.load_state_dict(saved_state)

    loaded_model.eval()

    check_inputs = torch.tensor([
        [ 2.0, 1.0],
        [-2.0, 0.0],
        [ 0.0, 3.0],
    ])

    with torch.no_grad():
        original_predictions = model(check_inputs)
        loaded_predictions = loaded_model(check_inputs)

    print("保存前：", original_predictions)
    print("加载后：", loaded_predictions)

    assert torch.allclose(
        original_predictions,
        loaded_predictions,
    ), "加载前后的预测不一致"

    print("通过：加载前后的预测一致")

if __name__ == "__main__":
    main()