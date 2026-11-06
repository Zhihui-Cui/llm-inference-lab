import torch
from torch import nn


def split_heads(x, num_heads):
    """[位置, 总特征] → [头, 位置, 头内特征]。"""
    T, D = x.shape
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError(f"头数必须是正整数，收到 {num_heads!r}")
    if D % num_heads != 0:
        raise ValueError(f"每个位置有 {D} 个特征，无法均分给 {num_heads} 个头")

    d = D // num_heads
    split_x = x.reshape(T, num_heads, d)
    return split_x.transpose(0, 1)


def merge_heads(x):
    """[头, 位置, 头内特征] → [位置, 总特征]。"""
    position_x = x.transpose(0, 1)
    T, num_heads, d = position_x.shape
    return position_x.reshape(T, num_heads * d)


num_heads = 2

queries = torch.tensor([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.],
])
head_queries = split_heads(queries, num_heads)

# 用 Q 里的标记数字检查：每个数都应回到原来的位置和特征列。
assert torch.equal(merge_heads(head_queries), queries), "拆分后再合并，元素顺序有误"

keys = torch.tensor([
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 1., 1.],
])
head_keys = split_heads(keys, num_heads)

# 每个头各算一张表。第 i 行由位置 i 的 Q 发起，第 j 列使用位置 j 的 K。
# [头, 查询位置, 头内特征] @ [头, 头内特征, 来源位置]
# → [头, 查询位置, 来源位置]
head_scores = head_queries @ head_keys.transpose(-2, -1)
scaled_scores = head_scores / (head_queries.shape[-1] ** 0.5)

sequence_length = head_queries.shape[-2]
allowed = torch.ones(
    sequence_length,
    sequence_length,
    dtype=torch.bool,
    device=head_queries.device,
).tril()
masked_scores = scaled_scores.masked_fill(~allowed, float("-inf"))
head_weights = torch.softmax(masked_scores, dim=-1)

# 对照读同一头的同一行。默认两头时，头 2、位置 2 的 Q 是 [7, 8]。
# 它对三个 K 的分数是 [7, 8, 15]；位置 3 被遮住，比例约为 [0.33, 0.67, 0]。
for head_index in range(num_heads):
    print(f"头 {head_index + 1} 的原始分数（行=谁在查，列=来源位置）：")
    print(head_scores[head_index])
    print(f"头 {head_index + 1} 的位置权重：")
    print(head_weights[head_index])

values = torch.tensor([
    [10., 0., 0., 20.],
    [20., 0., 0., 40.],
    [30., 0., 0., 60.],
])
head_values = split_heads(values, num_heads)

# 权重表第 j 列乘的是 V 的第 j 行，即同一来源位置提供的整条头内向量。
head_outputs = head_weights @ head_values
merged_outputs = merge_heads(head_outputs)
print("各头的输出：")
print(head_outputs)
print("按位置拼接后的输出：")
print(merged_outputs)

# 同一个线性层逐位置组合各头的特征。
model_width = merged_outputs.shape[-1]
output_layer = nn.Linear(model_width, model_width)
final_outputs = output_layer(merged_outputs)
print("输出投影后的形状：", final_outputs.shape)

reference_outputs = torch.nn.functional.scaled_dot_product_attention(
    head_queries,
    head_keys,
    head_values,
    allowed,
    dropout_p = 0.0
)

# 第一份：切输出。从“我爱学”已经算好的结果中，取出“我爱”的部分。
expected_short = head_outputs[:, :2, :]

# 第二份：切输入。现在只给两个位置的 Q、K、V，并使用 2×2 的掩码。
q_short = head_queries[:, :2, :]
k_short = head_keys[:, :2, :]
v_short = head_values[:, :2, :]
allowed_short = allowed[:2, :2]

# 用这两个位置重新算一次你熟悉的 Attention。
scores = q_short @ k_short.transpose(-2, -1)
scores = scores / (q_short.shape[-1] ** 0.5)
weights = torch.softmax(scores.masked_fill(~allowed_short, float("-inf")), dim=-1)
actual_short = weights @ v_short

torch.testing.assert_close(actual_short, expected_short, rtol=1e-5, atol=1e-6)

reference_outputs_short = torch.nn.functional.scaled_dot_product_attention(
    q_short,
    k_short,
    v_short,
    allowed_short,
    dropout_p = 0.0
)

# 输入都是“我爱”，换实现：手写结果和 PyTorch 对得上吗？
torch.testing.assert_close(actual_short, reference_outputs_short, rtol=1e-5, atol=1e-6)

# 都用手写实现，换长度：删掉“学”后，“我爱”的输出还与原来一致吗？
torch.testing.assert_close(actual_short, expected_short, rtol=1e-5, atol=1e-6)

q_batch = torch.stack([head_queries, head_queries], dim=0)
k_batch = torch.stack([head_keys, head_keys], dim=0)
v_batch = torch.stack([head_values, head_values + 10], dim=0)

scores_batch = q_batch @ k_batch.transpose(-2, -1)
scores_batch = scores_batch / (q_batch.shape[-1] ** 0.5)
weights_batch = torch.softmax(scores_batch.masked_fill(~allowed, float("-inf")), dim=-1)
batch_outputs = weights_batch @ v_batch

