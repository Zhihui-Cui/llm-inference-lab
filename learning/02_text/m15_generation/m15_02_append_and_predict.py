"""M15·2｜把第一个续写字加回同一句话，让模型再预测一个字。"""

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
    with torch.no_grad():
        scores = model(input_ids)
        next_scores = scores[0, -1]
        next_id = next_scores.argmax().reshape(1, 1)
        print("第一轮输入", input_ids.tolist(), "选出的编号", next_id.tolist())

        input_ids = torch.cat((input_ids, next_id), dim=-1)
        print("第二轮输入", input_ids.tolist(), decode(input_ids[0].tolist(), id_to_char))
        scores = model(input_ids)
        print("第二轮分数形状", list(scores.shape))
        fourth_id = scores[0, -1].argmax().item()
        print("第四个字", id_to_char[fourth_id])


if __name__ == "__main__":
    main()
