import torch
from torch import nn

queries = torch.tensor([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.],
])

# [位置, 特征] → [位置, 头, 每头特征]
split_queries = queries.reshape(3, 2, 2)

# [位置, 头, 每头特征] → [头, 位置, 每头特征]
head_queries = split_queries.transpose(0, 1)

# 练习用的 K：每行对应一个位置。
# 和当前 Q 一样，这里直接给定数字，暂时没有训练线性层。
keys = torch.tensor([
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 1., 1.],
])

# 与 Q 一样，把 K 拆成两个头。
# [位置, 总特征] → [位置, 头, 每头特征]
split_keys = keys.reshape(3, 2, 2)

# → [头, 位置, 每头特征]
head_keys = split_keys.transpose(0, 1)

# 每个头用自己的 Q 与自己的 K 计算匹配分数。
# [2, 3, 2] @ [2, 2, 3] → [2, 3, 3]
head_scores = head_queries @ head_keys.transpose(-2, -1)

print("头 1 的分数：")
print(head_scores[0])

print("头 2 的分数：")
print(head_scores[1])

# 每个头按自己的 Q、K 向量维度缩放。
scaled_scores = head_scores / (head_queries.shape[-1] ** 0.5)

# 三个位置的因果规则，两个头共同使用。
sequence_length = head_queries.shape[-2]
allowed = torch.ones(
    sequence_length,
    sequence_length,
    dtype=torch.bool,
    device=head_queries.device,
).tril()

# [3, 3] 的规则会应用到两个头各自的分数矩阵。
masked_scores = scaled_scores.masked_fill(~allowed, float("-inf"))

# 每个头、每个查询位置，分别计算一行取用比例。
head_weights = torch.softmax(masked_scores, dim=-1)

# 练习用的 V：每行是一个位置提供的信息。
values = torch.tensor([
    [10., 0., 0., 20.],
    [20., 0., 0., 40.],
    [30., 0., 0., 60.],
])

# 把 V 拆成两个头，与 Q、K 的拆法一致。
split_values = values.reshape(3, 2, 2)
head_values = split_values.transpose(0, 1)

head_outputs = head_weights @ head_values

# [头, 位置, 每头特征] → [位置, 头, 每头特征]
position_outputs = head_outputs.transpose(0, 1)

# 把同一个位置的两个头拼起来：[3, 2, 2] → [3, 4]
merged_outputs = position_outputs.reshape(3, 4)

print("两个头各自的输出：")
print(head_outputs)

print("按位置拼接后的输出：")
print(merged_outputs)

# 每个位置有四个拼接后的特征。
# 线性层分别处理每个位置，输出四个新特征。
output_layer = nn.Linear(4, 4)

final_outputs = output_layer(merged_outputs)

print("输出投影后的形状：", final_outputs.shape)