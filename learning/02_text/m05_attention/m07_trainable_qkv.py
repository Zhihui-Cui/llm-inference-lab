import torch
from torch import nn

def make_qkv(inputs, query_layer, key_layer, value_layer):
    queries = query_layer(inputs)
    keys = key_layer(inputs)
    values = value_layer(inputs)

    return queries, keys, values

def causal_attention(queries, keys, values):
    # 1. 计算所有查询与所有键的匹配分数。
    # 提示：只交换 keys 的最后两个维度。
    scores = queries @ keys.transpose(-2, -1)

    # 2. 根据 Q、K 的向量维度缩放。
    scaled_scores = scores / (queries.shape[-1] ** 0.5)

    # 3. 你刚才已经判断正确：倒数第二维是位置数。
    sequence_length = queries.shape[-2]

    # 4. 把现有代码中“创建下三角 allowed”的部分搬到这里。
    # 设备使用 queries.device，函数不再依赖外面的 inputs。
    allowed = torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
        device=queries.device,
    ).tril()

    # 5. 把禁止位置的分数替换为负无穷。
    masked_scores = scaled_scores.masked_fill(
        ~allowed,
        float("-inf"),
    )

    # 6. 计算组合比例。
    weights = torch.softmax(masked_scores, dim=-1)

    # 7. 按比例组合 V。
    outputs = weights @ values

    return outputs, weights

torch.manual_seed(42)

# 给定三个位置的输入向量。
# 每行是一个位置，每个位置有两个数字。
# 这里直接给定向量，暂时不接文本和 Embedding。
inputs = torch.tensor([
    [2.0, 1.0],
    [1.0, 2.0],
    [1.0, 1.0],
])

# 创建三个独立的线性层。
# 每个层都把两个输入数字转换成两个输出数字。
# 暂时不使用偏置，便于观察权重。
query_layer = nn.Linear(2, 2, bias=False)
key_layer = nn.Linear(2, 2, bias=False)
value_layer = nn.Linear(2, 2, bias=False)

# 同一份输入，分别经过三套不同参数。
queries, keys, values = make_qkv(inputs, query_layer, key_layer, value_layer)

# 查看各层的参数。
print("查询层的权重：")
print(query_layer.weight.detach())

print("键层的权重：")
print(key_layer.weight.detach())

print("值层的权重：")
print(value_layer.weight.detach())

# 查看这些参数实际算出的结果。
print("输入向量：")
print(inputs)

print("计算出的 Q：")
print(queries.detach())

print("计算出的 K：")
print(keys.detach())

print("计算出的 V：")
print(values.detach())

attention_outputs, attention_weights = causal_attention(queries, keys, values)

print("Attention 权重：")
print(attention_weights.detach())

print("Attention 输出：")
print(attention_outputs.detach())

# 给最后一个位置设置一个用于检查梯度的目标。
target_output = torch.tensor(
    [1.0, 0.0],
    dtype=inputs.dtype,
    device=inputs.device,
)

# 使用刚才实际计算出的 Attention 输出。
last_output = attention_outputs[-1]

loss_fn = nn.MSELoss()
loss = loss_fn(last_output, target_output)

print("当前最后一个位置的输出：", last_output.detach())
print("检查用目标：", target_output)
print("损失：", loss.item())

# 10. 从这个损失开始，计算相关参数的梯度。
loss.backward()

print("查询层权重的梯度：")
print(query_layer.weight.grad)

print("键层权重的梯度：")
print(key_layer.weight.grad)

print("值层权重的梯度：")
print(value_layer.weight.grad)

# 11. 把三个层的参数都交给同一个优化器。
# list(...) 取出每个层的参数，+ 把三个参数列表连接起来。
parameters = (
    list(query_layer.parameters())
    + list(key_layer.parameters())
    + list(value_layer.parameters())
)

optimizer = torch.optim.SGD(parameters, lr=0.03)

# 保存查询层更新前的权重，方便对照。
query_weight_before = query_layer.weight.detach().clone()

# 使用刚才 backward() 算出的梯度，更新三个层。
optimizer.step()

print("查询层更新前的权重：")
print(query_weight_before)

print("查询层更新后的权重：")
print(query_layer.weight.detach())

# 12. 使用更新后的参数，重新计算。
# 这里只检查新结果，不需要再记录梯度。
with torch.no_grad():
    new_queries, new_keys, new_values = make_qkv(inputs, query_layer, key_layer, value_layer)

    new_outputs, new_attention_weights = causal_attention(new_queries, new_keys, new_values)

    new_loss = loss_fn(new_outputs[-1], target_output)

print("更新前的最后位置输出：", last_output.detach())
print("更新后的最后位置输出：", new_outputs[-1])
print("更新前的损失：", loss.item())
print("更新后的损失：", new_loss.item())

inputs_batch = torch.stack(
    [inputs, inputs + 1],
    dim=0,
)

print("两份序列：")
print(inputs_batch)
print("批次输入形状：", inputs_batch.shape)

queries_batch, keys_batch, values_batch = make_qkv(inputs_batch, query_layer, key_layer, value_layer)

print("Q 形状：", queries_batch.shape)
print("K 形状：", keys_batch.shape)
print("V 形状：", values_batch.shape)

outputs_batch, weights_batch = causal_attention(queries_batch, keys_batch, values_batch)

print("权重形状：", weights_batch.shape)
print("输出形状：", outputs_batch.shape)

print("每份序列、每个查询的权重总和：")
print(weights_batch.sum(dim=-1).detach())

# 1. 保留原结果作为对照。
original_outputs = outputs_batch.detach().clone()

# 2. 复制输入，只修改第二份序列。
changed_inputs = inputs_batch.clone()
changed_inputs[1] += 10

# 3. 使用相同参数重新计算，期间不更新参数。
with torch.no_grad():
    changed_q, changed_k, changed_v = make_qkv(changed_inputs, query_layer, key_layer, value_layer)

    changed_outputs, changed_weights = causal_attention(changed_q, changed_k, changed_v)

# 4. 分别比较两份序列。
print("第一份序列的输出变化：")
print(changed_outputs[0] - original_outputs[0])

print("第二份序列的输出变化：")
print(changed_outputs[1] - original_outputs[1])

assert torch.isfinite(original_outputs).all()
assert torch.isfinite(changed_outputs).all()

# 第一份输入没有改变，输出也应该不变。
assert torch.allclose(
    changed_outputs[0],
    original_outputs[0],
    rtol=1e-5,
    atol=1e-6,
)

# 本例第二份输入发生了明显变化，输出也应改变。
assert not torch.allclose(
    changed_outputs[1],
    original_outputs[1],
    rtol=1e-5,
    atol=1e-6,
)

print("通过：修改第二份序列，没有影响第一份序列的输出")