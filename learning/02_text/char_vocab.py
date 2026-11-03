train_text = "我爱学习，我爱代码。代码帮助我学习。"
validation_text = "我爱猫。"
test_text = "我爱狗。"

# 去重后排序，让字符顺序固定。
chars = sorted(set(train_text))

# 预留 0，表示未知字符。
char_to_id = {"<unk>": 0}

# 同时获取编号和字符，从 1 开始编号。
for index, char in enumerate(chars, start=1):
    char_to_id[char] = index

id_to_char = {}  # 编号 → 字符，用于解码

for char, index in char_to_id.items():
    id_to_char[index] = char

def encode(text, char_to_id):
    ids = []

    for char in text:
        index = char_to_id.get(char, char_to_id["<unk>"])

        ids.append(index)

    return ids

def decode(ids, id_to_char):
    chars = []

    for index in ids:
        chars.append(id_to_char[index])

    return "".join(chars)

if __name__ == "__main__":
    # 训练文本中的所有字符，都应能够完整往返。
    assert decode(encode(train_text, char_to_id), id_to_char) == train_text

    # 验证和测试中的新字符没有进入训练词表。
    assert "猫" not in char_to_id
    assert "狗" not in char_to_id

    # 未知字符统一编码为 <unk>，解码时不能恢复原字符。
    assert decode(
        encode(validation_text, char_to_id), id_to_char
    ) == "我爱<unk>。"

    assert decode(
        encode(test_text, char_to_id), id_to_char
    ) == "我爱<unk>。"

    print("通过：已知字符往返一致，未知字符处理符合预期")