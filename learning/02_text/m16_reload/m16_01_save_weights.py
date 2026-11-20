"""M16·1｜保存四轮训练后的参数，从文件读回同一张位置表。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m15_generation.m15_01_next_character import prepare_model


def main():
    model = prepare_model()
    weights_path = Path(__file__).resolve().parent / "artifacts" / "m14_four_epochs.pth"
    weights_path.parent.mkdir(exist_ok=True)
    before = model.position_embedding.weight[0, 0].item()

    torch.save(model.state_dict(), weights_path)

    saved_weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    saved_table = saved_weights["position_embedding.weight"]
    print("保存位置", weights_path)
    print("模型里的位置表 [0, 0]", before)
    print("从文件读回的位置表 [0, 0]", saved_table[0, 0].item())
    print("整张位置表完全相同", torch.equal(model.position_embedding.weight, saved_table))


if __name__ == "__main__":
    main()
