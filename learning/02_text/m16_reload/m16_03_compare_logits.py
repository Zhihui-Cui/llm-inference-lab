"""M16·3｜同一个“我爱”：四轮训练得到的参考模型，与从文件重载的模型，词表分数是否一致？"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder
from char_vocab import char_to_id, encode
from m15_generation.m15_01_next_character import prepare_model


def main():
    input_ids = torch.tensor([encode("我爱", char_to_id)], dtype=torch.long)

    # 只为这次对照：按第一课同样的种子、文本和顺序复现四轮训练，取得参考分数。
    reference_model = prepare_model()
    with torch.no_grad():
        reference_scores = reference_model(input_ids)

    # 这是以后实际使用的路线：创建模型、读文件、装参数。
    loaded_model = TinyDecoder()
    weights_path = Path(__file__).resolve().parent / "artifacts" / "m14_four_epochs.pth"
    saved_weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    loaded_model.load_state_dict(saved_weights)
    loaded_model.eval()
    with torch.no_grad():
        loaded_scores = loaded_model(input_ids)

    print("同一个输入", input_ids.tolist(), "两组分数形状", list(reference_scores.shape), list(loaded_scores.shape))
    max_difference = (reference_scores - loaded_scores).abs().max().item()
    print("全部位置、全部候选字的最大绝对差", max_difference)
    torch.testing.assert_close(loaded_scores, reference_scores, rtol=1e-5, atol=1e-6)
    print("重载分数核对通过")


if __name__ == "__main__":
    main()
