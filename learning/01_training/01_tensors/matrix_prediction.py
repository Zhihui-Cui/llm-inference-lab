import torch

# 每一行是一个样本，每一列是一个特征。
inputs = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

weights = torch.tensor([
    [2.0],
    [3.0],
])

bias = torch.tensor(1.0)

# 用 inputs @ weights + bias 计算 predictions
predictions = inputs @ weights + bias

# 打印 inputs、weights、predictions 的 shape
print("inputs.shape:", inputs.shape)
print("weights.shape:", weights.shape)
print("predictions.shape:", predictions.shape)

# 打印 predictions
print("predictions:", predictions)