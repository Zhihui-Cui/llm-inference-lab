import torch
from torch import nn

loss_fn = nn.CrossEntropyLoss()

# 只有一个包裹，真实类别是中件，编号 1。
targets = torch.tensor([1], dtype=torch.long)

# A：最高分给了轻件，预测错误。
logits_a = torch.tensor([[3.0, 1.0, 0.0]])

# B：最高分给了中件，预测正确。
logits_b = torch.tensor([[1.0, 3.0, 0.0]])

loss_a = loss_fn(logits_a, targets)
loss_b = loss_fn(logits_b, targets)

print("A 的预测类别：", logits_a.argmax(dim=1))
print("A 的损失：", loss_a.item())

print("B 的预测类别：", logits_b.argmax(dim=1))
print("B 的损失：", loss_b.item())