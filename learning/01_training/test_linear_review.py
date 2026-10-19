from pathlib import Path

import torch
from torch import nn


def main() -> None:
    # 1. 创建相同结构，并加载已经选好的参数。
    model_path = Path(__file__).with_name("linear_review_weights.pth")

    model = nn.Linear(1, 1)
    saved_state = torch.load(
        model_path,
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(saved_state)
    model.eval()

    # 2. 新的测试样本：
    # 不与本次线性练习的训练集、验证集重复。
    # 这些输入都位于训练范围 [-2, 2] 内。
    test_inputs = torch.tensor([
        [-1.8],
        [-0.6],
        [ 0.2],
        [ 1.8],
    ])

    # 真实关系：y = -2x + 3。
    test_targets = torch.tensor([
        [ 6.6],
        [ 4.2],
        [ 2.6],
        [-0.6],
    ])

    # 3. 只预测和统计，不执行参数更新。
    with torch.no_grad():
        predictions = model(test_inputs)

        errors = predictions - test_targets
        test_mse = (errors ** 2).mean().item()
        max_absolute_error = errors.abs().max().item()

    # 4. 查看每个样本的预测。
    print("输入 | 真实值 | 预测值")
    for index in range(len(test_inputs)):
        print(
            f"{test_inputs[index].item():5.1f} | "
            f"{test_targets[index].item():7.3f} | "
            f"{predictions[index].item():10.6f}"
        )

    print(f"\n测试 MSE：{test_mse:.6e}")
    print(f"最大绝对误差：{max_absolute_error:.6e}")
    print("是否每个测试样本的误差都小于 0.05：",
          max_absolute_error < 0.05)


if __name__ == "__main__":
    main()