import torch

predictions = torch.tensor([[1.0], [2.0], [3.0]])
targets = torch.tensor([1.0, 2.0, 3.0])

# 故障：形状不同，触发广播。
difference = predictions - targets
print("修复前的差值：", difference)
print("修复前 shape：", difference.shape)
print("修复前 MSE：", (difference ** 2).mean().item())

# 修复：让每个预测值只与对应标签比较。
targets = targets.reshape(-1, 1)

difference = predictions - targets
print("修复后的差值：", difference)
print("修复后 shape：", difference.shape)
print("修复后 MSE：", (difference ** 2).mean().item())

assert predictions.shape == targets.shape
assert torch.allclose(predictions, targets)
print("通过：形状一致，预测值与标签一致")