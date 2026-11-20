import torch
from torch import nn

def attend_one_position(q, keys, values):
    # 当前位置的 Q，与各个位置的 K 计算分数。
    scores = (keys * q).sum(dim=1)

    # 根据 Q、K 的向量维度缩放。
    scaled_scores = scores / (q.shape[0] ** 0.5)

    # 在所有被关注的位置之间分配权重。
    weights = torch.softmax(scaled_scores, dim=0)

    # 累加每个位置提供的信息，不再写死只有三个位置。
    output = torch.zeros_like(values[0])

    for index in range(len(values)):
        output += weights[index] * values[index]

    return output, weights

# 三个位置的 K，每行对应一个位置。
keys = torch.tensor([
    [2.0, 1.0],
    [1.0, 2.0],
    [1.0, 1.0],
])

# 三个位置的 V，每行对应一个位置。
values = torch.tensor([
    [4.0, 3.0],
    [2.0, 6.0],
    [2.0, 3.0],
])

queries = torch.tensor([
    [3.0, 1.0],
    [3.0, -1.0],
    [2.0, 0.0],
])

for position, current_q in enumerate(queries, start=1):
    current_output, current_weights = attend_one_position(
        current_q, keys, values
    )

    print(f"第 {position} 个位置的权重：", current_weights)
    print(f"第 {position} 个位置的输出：", current_output)

# keys.T 把原来每一行的 K 转成每一列。
print("转置后的 K：")
print(keys.T)

# @ 表示矩阵乘法，不是逐个元素相乘。
# 它在这里一次计算出所有 Q 与所有 K 的点积。
all_scores = queries @ keys.T

print("所有位置的匹配分数：")
print(all_scores)

# 1. 每个分数除以 Q、K 向量维度的平方根。
# queries.shape 是 [3, 2]，最后一个维度是 2。
all_scaled_scores = all_scores / (queries.shape[-1] ** 0.5)

# 2. 每一行分别做 softmax。
all_weights = torch.softmax(all_scaled_scores, dim=-1)

print("所有位置的权重：")
print(all_weights)

print("每一行的权重总和：")
print(all_weights.sum(dim=-1))

all_outputs = all_weights @ values

print("所有位置的输出：")
print(all_outputs)

for position, current_q in enumerate(queries):
    expected_output, expected_weights = attend_one_position(
        current_q, keys, values
    )

    assert torch.allclose(
        all_weights[position], expected_weights,
        rtol=1e-5, atol=1e-6,
    )
    assert torch.allclose(
        all_outputs[position], expected_output,
        rtol=1e-5, atol=1e-6,
    )

print("通过：矩阵计算与逐位置计算一致")

def single_head_attention(queries, keys, values):
    # 1. 所有 Q 与所有 K 计算点积，得到分数矩阵。
    scores = queries @ keys.T

    # 2. 根据 Q、K 的向量维度缩放分数。
    scaled_scores = scores / (queries.shape[-1] ** 0.5)

    # 3. 对每一行计算 softmax，得到权重。
    weights = torch.softmax(scaled_scores, dim=-1)

    # 4. 按权重组合 V，得到所有位置的输出。
    outputs = weights @ values

    return outputs, weights

outputs, weights = single_head_attention(
    queries, keys, values
)

print("函数输出：", outputs)
print("函数权重：", weights)

assert torch.allclose(outputs, all_outputs, rtol=1e-5, atol=1e-6)
assert torch.allclose(weights, all_weights, rtol=1e-5, atol=1e-6)

print("通过：封装后的函数与原计算一致")

def causal_attention(queries, keys, values):
    scores = queries @ keys.T
    scaled_scores = scores / (queries.shape[-1] ** 0.5)

    # 本次 Q、K 来自同一个序列，位置数量相同。
    sequence_length = queries.shape[0]

    # True 表示允许关注：保留对角线及其下方的位置。
    allowed = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=scores.device,
    ).tril()

    # ~allowed 将允许/禁止翻转。
    # masked_fill 在 True 的位置填入指定值，
    # 因此这里把禁止位置的分数替换为负无穷。
    masked_scores = scaled_scores.masked_fill(
        ~allowed, float("-inf")
    )

    weights = torch.softmax(masked_scores, dim=-1)
    outputs = weights @ values

    return outputs, weights

causal_outputs, causal_weights = causal_attention(
    queries, keys, values
)

print("因果 Attention 权重：")
print(causal_weights)

print("因果 Attention 输出：")
print(causal_outputs)

# 1. 复制数据，只改变第 3 个位置的信息。
changed_keys = keys.clone()
changed_values = values.clone()

changed_keys[2] += 10
changed_values[2] += 10

# 2. 比较加掩码时，修改前后的输出。
before, _ = causal_attention(queries, keys, values)
after, _ = causal_attention(
    queries, changed_keys, changed_values
)

print("加掩码：前两个位置的输出变化")
print(after[:2] - before[:2])

assert torch.allclose(
    before[:2], after[:2],
    rtol=1e-5, atol=1e-6,
)

# 3. 用不加掩码的函数做对照。
plain_before, _ = single_head_attention(
    queries, keys, values
)
plain_after, _ = single_head_attention(
    queries, changed_keys, changed_values
)

print("不加掩码：前两个位置的输出变化")
print(plain_after[:2] - plain_before[:2])

assert not torch.allclose(
    plain_before[:2], plain_after[:2],
    rtol=1e-5, atol=1e-6,
)

print("通过：掩码阻止了未来位置的信息影响前面的输出")

# 1. 使用之前第 3 个位置的 Attention 输出。
# 一份样本、两个特征，shape 是 [1, 2]。
features = torch.tensor([[3.3457, 3.4907]])

# 2. 暂时只有两个候选字符。
candidates = ["习", "猫"]

# 正确答案是“习”，对应候选编号 0。
target = torch.tensor([0], dtype=torch.long)

# 3. 把两个输入特征，转换成两个候选字符的分数。
torch.manual_seed(42)
output_layer = nn.Linear(2, 2)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(output_layer.parameters(), lr=0.1)

# 4. 训练这个分类层。
for step in range(21):
    scores = output_layer(features)
    loss = loss_fn(scores, target)

    if step in (0, 20):
        probabilities = torch.softmax(scores, dim=-1)
        print(f"更新 {step} 次后的损失：{loss.item():.4f}")
        print("“习”和“猫”的概率：", probabilities.detach())

    if step < 20:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# [:1] 只取第一行，同时保留二维形状。
# Q、K、V 的形状都是 [1, 2]。
one_query = queries[:1]
one_key = keys[:1]
one_value = values[:1]

one_output, one_weight = causal_attention(
    one_query, one_key, one_value
)

print("只有一个位置时的权重：", one_weight)
print("只有一个位置时的输出：", one_output)

# 结果不能出现 NaN 或无穷大。
assert torch.isfinite(one_weight).all()
assert torch.isfinite(one_output).all()

# 唯一允许关注的位置就是自己，所以权重为 1。
assert torch.allclose(
    one_weight,
    torch.ones_like(one_weight),
)

# 1 × 自己的 V，结果应等于自己的 V。
assert torch.allclose(one_output, one_value)

print("通过：只有一个位置时，权重为 1，输出等于自己的 V")