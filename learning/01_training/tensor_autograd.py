import torch

# 我的预测：
# x = 3 时，y = 15
# x = 3 时，dy/dx = 8

x = torch.tensor(3.0, requires_grad=True)

# y = x² + 2x
y = x ** 2 + x * 2

# 打印 y
print(y)

# 执行 y.backward()
y.backward()

# 打印 x.grad

print(x.grad)

# 打印 x，检查它有没有改变
print(x)

learning_rate = 0.1

with torch.no_grad():
    # 按更新公式修改 x
    x -= learning_rate * x.grad

# 用更新后的 x 重新计算 y_new
y_new = x ** 2 + x * 2

# 打印 x 和 y_new
print(x)
print(y_new)

y_new.backward()
print("第二次 backward 后的梯度：", x.grad)

x.grad.zero_()
print("清零后的梯度：", x.grad)
print("清零后的参数：", x)