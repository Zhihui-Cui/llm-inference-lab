import torch

y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([2.0, 2.0, 5.0])

# 计算差值
difference = y_true - y_pred
# 计算差值的平方
squared_error = difference ** 2
# 计算平方误差的平均值
loss = squared_error.mean()

print("差值：", difference, "shape：", difference.shape)
print("平方误差：", squared_error, "shape：", squared_error.shape)
print("MSE：", loss, "shape：", loss.shape)