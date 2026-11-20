"""M11｜三个位置一起经过 decoder 时，前两个位置能看到未来吗？
先预测去掉第三个位置后前两行会不会变，再运行本文件检查。
"""

import torch
from torch import nn


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


def check_prefix(block):
    # 单句练习：手工给三个位置各两个数，用“我、爱、学”给三行标记位置。
    x = torch.tensor([
        [2., 4.],
        [14., 10.],
        [3., 6.],
    ])

    y_full = block(x)          # 三个位置一起算，输出 [3,2]
    y_short = block(x[:2])     # [:2] 只取前两个位置，输出 [2,2]
    assert y_full.shape == x.shape and y_short.shape == x[:2].shape
    # 同一套参数下，裁掉未来位置，前两个位置的输出应在容差内一致。
    torch.testing.assert_close(y_full[:2], y_short, rtol=1e-5, atol=1e-6)
    return y_full


if __name__ == "__main__":
    block = DecoderBlock(d_model=2, d_ff=4, num_heads=2)
    y_full = check_prefix(block)
    print("三个位置的输出形状", y_full.shape, "裁掉第三个位置后，前两行核对通过")
