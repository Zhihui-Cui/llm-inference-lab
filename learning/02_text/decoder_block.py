"""M11–M12：用 decoder block 处理各位置的向量，再用下一个字的编号计算损失。"""

import torch
from torch import nn

from char_vocab import char_to_id, encode


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        # 这里创建的层属于这个 block。长句、短句和 batch 都会使用同一套参数。
        # LayerNorm(d_model) 在每个位置内部，沿最后一维的 D 个特征做归一化。
        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,  # 三维输入按 [句子数 B, 位置数 T, 特征数 D] 读取
        )
        # FFN 分支有自己的 LayerNorm。两个线性层逐位置处理特征，D → d_ff → D。
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        # x 可以是一句 [T,D]，也可以是一批 [B,T,D]。两者的倒数第二维都是 T。
        z_attn = self.norm_attn(x)
        T = x.shape[-2]

        # 行 i 是发起查询的位置，列 j 是被读取的位置。triu(1) 选中对角线上方。
        # MultiheadAttention 的布尔掩码中 True 表示禁止。T=3 时第二行为 [False,False,True]。
        blocked = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        # 同一个 z_attn 分别送进 Q/K/V 投影。a 是 Attention 做完输出投影后的结果。
        # need_weights=False 表示只用输出；用 _ 接住返回值中的第二项。
        a, _ = self.attention(
            z_attn, z_attn, z_attn, attn_mask=blocked, need_weights=False
        )

        h = x + a  # 第一次残差：原输入 + Attention 分支输出
        z = self.norm_ffn(h)
        f = self.ffn(z)  # 每行只加工这个位置的特征，各行共用 FFN 参数
        y = h + f  # 第二次残差从 h 出发。y 仍是各位置的表示。
        return y


# 单句练习：手工给三个位置各两个数，用“我、爱、学”给三行标记位置。
x = torch.tensor([
    [2., 4.],
    [14., 10.],
    [3., 6.],
])
block = DecoderBlock(d_model=2, d_ff=4, num_heads=2)

y_full = block(x)          # 三个位置一起算，输出 [3,2]
y_short = block(x[:2])     # [:2] 只取前两个位置，输出 [2,2]
assert y_full.shape == x.shape and y_short.shape == x[:2].shape
# 同一套参数下，裁掉未来位置，前两个位置的输出应在容差内一致。
torch.testing.assert_close(y_full[:2], y_short, rtol=1e-5, atol=1e-6)

# 同一个预测层逐行打分。每行两个特征 → 每个候选字一个原始分数。
lm_head = nn.Linear(y_full.shape[-1], len(char_to_id))
logits = lm_head(y_full)  # [3,11]；logits[i,j] 是位置 i 给编号 j 的分数
print(logits)

# 原文“我爱学习”的三个下一字依次是“爱、学、习”。targets[i] 指定第 i 行的正确列。
# 列表推导按顺序查三个字的 ID；long 表示整数编号。targets 的形状是 [3]。
targets = torch.tensor(
    [char_to_id[ch] for ch in "爱学习"],
    dtype=torch.long,
    device=logits.device,
)

# 单独检查位置 2。[1:2] 从下标 1 开始，取到下标 2 之前，保留“一行”的维度。
# [1,11] 的分数配 [1] 的答案，得到这一个位置的损失。
loss_fn = nn.CrossEntropyLoss()
loss_pos2 = loss_fn(logits[1:2], targets[1:2])

# 把刚才的计算展开来观察。[1] 取出位置 2 的 11 个分数，
# dim=-1 沿候选字的维度计算概率；targets[1] 再选中“学”对应的那一列。
probs_pos2 = torch.softmax(logits[1], dim=-1)
p_xue = probs_pos2[targets[1]]
print("读到“我爱”时，“学”的概率：", p_xue.item())
# .item() 把单个数值取成 Python 数字，torch.log 是自然对数。
print("用这个概率手算的损失：", -torch.log(p_xue).item())
print("loss_pos2：", loss_pos2.item())

# 交叉熵接收原始 logits。默认把三个位置的损失取平均，返回一个数。
loss = loss_fn(logits, targets)
# reduction="none" 保留三个损失。先创建这个损失函数，再用括号传入分数和答案。
per_position_loss_fn = nn.CrossEntropyLoss(reduction="none")
losses_by_pos = per_position_loss_fn(logits, targets)
print("三个位置各自的损失：", losses_by_pos.tolist())  # .tolist() 方便逐项阅读
print("三个数的平均：", losses_by_pos.mean().item(), "整句 loss：", loss.item())


# 批次练习：这次从真实字符 ID 开始。两个句子各四个字，ids_batch 是 [2,4]。
ids_batch = torch.tensor([
    encode("我爱学习", char_to_id),
    encode("代码帮助", char_to_id),
], dtype=torch.long)

# 逗号左边的 : 保留两句话，右边的切片在每句话内部取位置。
inputs_batch = ids_batch[:, :-1]  # [2, 3]，去掉末字 → “我爱学”、“代码帮”
targets_batch = ids_batch[:, 1:]  # [2, 3]，去掉首字 → “爱学习”、“码帮助”

token_embedding = nn.Embedding(len(char_to_id), 2)
position_embedding = nn.Embedding(8, 2)  # 最多 8 个位置，每个位置向量有 2 个特征
position_ids = torch.arange(inputs_batch.shape[1], device=inputs_batch.device)
token_vectors = token_embedding(inputs_batch)       # [2,3,2]，按字的编号查向量
position_vectors = position_embedding(position_ids)  # [3,2]，按位置 0、1、2 查向量
# 加法时两句话共用同一组位置向量。x_batch[1,0] = “代”的字符向量 + 位置 0 的向量。
x_batch = token_vectors + position_vectors

hidden_batch = block(x_batch)        # [2,3,2]，每句话只读取自己允许的上下文
logits_batch = lm_head(hidden_batch)  # [2,3,11]，每句话的每个位置各有 11 个分数

# 前两维合成“预测位置总数”。-1 让 PyTorch 算出 2×3=6，最后一维仍是词表大小。
logits_flat = logits_batch.reshape(-1, logits_batch.shape[-1])  # [6,11]
targets_flat = targets_batch.reshape(-1)                        # [6]，按同样顺序排列
loss_batch = loss_fn(logits_flat, targets_flat)  # 六个位置的平均损失

# 展平后的下标 3 是第 4 行，即第二句话的第一个位置：“代”预测“码”。
torch.testing.assert_close(logits_flat[3], logits_batch[1, 0])
assert targets_flat[3].item() == char_to_id["码"]

print("分数和目标的形状：", logits_flat.shape, targets_flat.shape)
print("六个位置的平均损失：", loss_batch.item())
