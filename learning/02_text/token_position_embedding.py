import torch
from torch import nn

from char_vocab import encode, char_to_id


def main():
    torch.manual_seed(42)

    text = "我爱学"
    ids = torch.tensor(
        encode(text, char_to_id),
        dtype=torch.long,
    )

    print("文字：", text)
    print("编号：", ids)  # [7, 8, 5]

    embedding = nn.Embedding(len(char_to_id), 2)

    max_positions = 8
    position_embedding = nn.Embedding(max_positions, 2)
    
    query_layer = nn.Linear(2, 2)
    key_layer = nn.Linear(2, 2)
    value_layer = nn.Linear(2, 2)

    T = ids.shape[0]
    position_ids = torch.arange(T, device=ids.device)
    token_vectors = embedding(ids)
    position_vectors = position_embedding(position_ids)
    x = token_vectors + position_vectors

    # swapped_ids = ids.flip(0)                  # “我爱学” → “学爱我”
    swapped_ids = ids.clone()
    swapped_ids[2] = ids[0]   # 把第 3 个字“学”换成“我”，得到“我爱我”
    swapped_tokens = embedding(swapped_ids)    # 用原来的字符表查
    swapped_x = swapped_tokens + position_vectors

    print(f"原字符向量：{token_vectors} | "
          f"新字符向量：{swapped_tokens} | "
          f"原向量：{x} | "
          f"新向量：{swapped_x}")
        
    
    # 同一个 x，分别送进三个不同的层。
    queries = query_layer(x)
    keys = key_layer(x)
    values = value_layer(x)

    swapped_queries = query_layer(swapped_x)
    swapped_keys = key_layer(swapped_x)
    swapped_values = value_layer(swapped_x)

    print("\n字符向量加位置向量后的输入 x：")
    print(x)

    print("\n查询层算出的 Q：")
    print(queries)

    print("\n键层算出的 K：")
    print(keys)

    print("\n值层算出的 V：")
    print(values)

    scores = queries @ keys.T
    scaled_scores = scores / (queries.shape[-1] ** 0.5)
    allowed = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    masked_scores = scaled_scores.masked_fill(~allowed, float("-inf"))
    attention_weights = torch.softmax(masked_scores, dim=-1)
    attention_outputs = attention_weights @ values

    swapped_scores = swapped_queries @ swapped_keys.T
    swapped_scaled_scores = swapped_scores / (swapped_queries.shape[-1] ** 0.5)
    allowed = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    swapped_masked_scores = swapped_scaled_scores.masked_fill(~allowed, float("-inf"))
    swapped_attention_weights = torch.softmax(swapped_masked_scores, dim=-1)
    swapped_attention_outputs = swapped_attention_weights @ swapped_values

    # 第一个位置只读取自己的 V；倒序实验中可检查是否误用了旧 V。
    torch.testing.assert_close(
        swapped_attention_outputs[0], swapped_values[0], rtol=1e-5, atol=1e-6
    )
    
    print("\nAttention 权重：")
    print(attention_weights)

    print("\nAttention 输出：")
    print(attention_outputs)

    print(f"原输入（位置 2）：{attention_outputs[1]} | "
          f"修改后（位置 2）：{swapped_attention_outputs[1]}")


if __name__ == "__main__":
    main()