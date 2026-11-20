"""M15·4｜最多续写三个字；前缀加续写的总编号数达到位置表容量就停止。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id, decode, encode, id_to_char
from m15_generation.m15_01_next_character import prepare_model


def main():
    model = prepare_model()
    prefix = "代码帮助我学习"
    input_ids = torch.tensor([encode(prefix, char_to_id)], dtype=torch.long)
    max_new_tokens = 3
    max_total_tokens = model.position_embedding.num_embeddings
    print("前缀", prefix, "前缀长度", input_ids.shape[1])
    print("最多新增", max_new_tokens, "含前缀的总长度上限", max_total_tokens)

    with torch.no_grad():
        for step in range(1, max_new_tokens + 1):
            current_length = input_ids.shape[1]
            if current_length == max_total_tokens:
                break
            print(f"第 {step} 次调用模型，送入长度", current_length)
            scores = model(input_ids)
            next_id = scores[0, -1].argmax().reshape(1, 1)
            input_ids = torch.cat((input_ids, next_id), dim=1)

    print("最终文本", decode(input_ids[0].tolist(), id_to_char))
    print("实际新增", input_ids.shape[1] - len(prefix), "最终编号数", input_ids.shape[1])


if __name__ == "__main__":
    main()
