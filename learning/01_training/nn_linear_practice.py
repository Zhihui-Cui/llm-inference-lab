import torch
from torch import nn

inputs = torch.tensor([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
])

model = nn.Linear(2, 1)

print("权重：", model.weight)
print("偏置：", model.bias)
print("权重是否需要梯度：", model.weight.requires_grad)

# 1：打印权重和偏置的 shape。
print(f"权重shape: {model.weight.shape}")
print(f"偏置shape: {model.bias.shape}")

# 2：用 model(inputs) 得到 predictions。
predictions = model(inputs)

# 3：用矩阵公式得到 manual_predictions。
manual_predictions = inputs @ model.weight.T + model.bias

# 4：打印两种预测，以及它们的 shape。
print(f"模型预测：{predictions}, 模型shape: {predictions.shape}")
print(f"手动预测：{manual_predictions}, 模型shape: {manual_predictions.shape}")

# 5：检查两种预测是否接近。
print(torch.allclose(predictions, manual_predictions))