"""M15·3｜从“我爱”开始，连续选 3 个新编号，每轮把完整的已有序列送进模型。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id, decode, encode, id_to_char
from m15_generation.m15_01_next_character import prepare_model


def main():
    model = prepare_model()
    input_ids = torch.tensor([encode("我爱", char_to_id)], dtype=torch.long)
    max_new_tokens = 3

    with torch.no_grad():
        for step in range(1, max_new_tokens + 1):
            print(f"第 {step} 轮送入", input_ids.tolist(), "输入长度", input_ids.shape[1])
            scores = model(input_ids)
            next_scores = scores[0, -1]
            next_id = next_scores.argmax().reshape(1, 1)
            input_ids = torch.cat((input_ids, next_id), dim=-1)
            print("本轮接上的字", id_to_char[next_id.item()])

    print("最终编号", input_ids.tolist())
    print("最终文本", decode(input_ids[0].tolist(), id_to_char))


if __name__ == "__main__":
    main()
