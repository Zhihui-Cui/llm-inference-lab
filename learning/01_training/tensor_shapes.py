import torch

targets = torch.tensor([3.0, 5.0, 7.0])

predictions = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
])

print("真实值的形状：", targets.shape)
print("预测值的形状：", predictions.shape)

targets = targets.reshape(3, 1)
difference = predictions - targets

print("difference:", difference, "difference.shape:",difference.shape)

print((difference ** 2).mean())