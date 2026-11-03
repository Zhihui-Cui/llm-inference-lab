import torch
from torch import nn

from char_vocab import encode, char_to_id


def main():
    torch.manual_seed(42)

    # ==================================================
    # 1. 准备输入：文字 → 编号
    # ==================================================
    text = "我爱学"

    ids = torch.tensor(
        encode(text, char_to_id),
        dtype=torch.long,
    )

    print("文字：", text)
    print("编号：", ids)  # [7, 8, 5]

    # ==================================================
    # 2. 创建各层：此处创建参数，还没有计算输入
    # ==================================================

    # 一张表：词表中每个编号对应一行，每行两个数。
    # 表里的数目前使用随机初始化值。
    embedding = nn.Embedding(len(char_to_id), 2)

    # 三个独立的线性层，各自有自己的权重和偏置。
    # 每个层：输入两个数，输出两个数。
    query_layer = nn.Linear(2, 2)
    key_layer = nn.Linear(2, 2)
    value_layer = nn.Linear(2, 2)

    # ==================================================
    # 3. 使用各层：实际计算 x、Q、K、V
    # ==================================================

    # 用编号查表。
    # 三个字符各取一行，x 的形状是 [3, 2]。
    x = embedding(ids)

    # 同一个 x，分别送进三个不同的层。
    queries = query_layer(x)
    keys = key_layer(x)
    values = value_layer(x)

    print("\nEmbedding 查出的向量：")
    print(x)

    print("\n查询层算出的 Q：")
    print(queries)

    print("\n键层算出的 K：")
    print(keys)

    print("\n值层算出的 V：")
    print(values)

    # ==================================================
    # 4. 使用真正算出来的 Q、K、V，执行 Attention
    # ==================================================

    # 每个 Q 与每个 K 计算点积。
    scores = queries @ keys.T

    # 每个 Q、K 有两个元素，因此除以 √2。
    scaled_scores = scores / (queries.shape[-1] ** 0.5)

    # 每一行对应一个查询，对所有被关注位置分配权重。
    attention_weights = torch.softmax(
        scaled_scores,
        dim=-1,
    )

    # 按权重组合各个位置的 V。
    attention_outputs = attention_weights @ values

    print("\nAttention 权重：")
    print(attention_weights)

    print("\nAttention 输出：")
    print(attention_outputs)


if __name__ == "__main__":
    main()