"""M15·5｜同一个生成函数接收“我爱”和空字符串，观察哪个输入会进入模型。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id, decode, encode


def generate(model, prefix, max_new_tokens, *, vocab=char_to_id):
    if prefix == "":
        raise ValueError("前缀不能为空")

    prefix_ids = encode(prefix, vocab)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long)
    max_total_tokens = model.position_embedding.num_embeddings
    if len(prefix_ids) > max_total_tokens:
        raise ValueError(f"前缀超过总长度上限 {max_total_tokens}")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= max_total_tokens:
                break
            print("生成前向，输入编号", input_ids.tolist())
            scores = model(input_ids)
            next_id = scores[0, -1].argmax().reshape(1, 1)
            input_ids = torch.cat((input_ids, next_id), dim=1)

    new_ids = input_ids[0, len(prefix_ids):].tolist()
    id_to_char = {index: char for char, index in vocab.items()}
    return prefix + decode(new_ids, id_to_char)


def main():
    from m15_generation.m15_01_next_character import prepare_model

    model = prepare_model()
    for prefix in ("我爱", ""):
        print("本次前缀", repr(prefix))
        try:
            print("续写结果", generate(model, prefix, max_new_tokens=1))
        except ValueError as error:
            print("输入提示", error)


if __name__ == "__main__":
    main()
