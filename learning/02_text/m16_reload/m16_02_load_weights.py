"""M16·2｜在新进程里创建 TinyDecoder，把上一课的参数装进新模型。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder


def main():
    torch.manual_seed(42)
    model = TinyDecoder()
    print("新模型刚创建时 [0, 0]", model.position_embedding.weight[0, 0].item())

    weights_path = Path(__file__).resolve().parent / "artifacts" / "m14_four_epochs.pth"
    saved_weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    print("文件里保存的 [0, 0]", saved_weights["position_embedding.weight"][0, 0].item())
    print("刚读完文件，模型里的 [0, 0]", model.position_embedding.weight[0, 0].item())
    model.load_state_dict(saved_weights)
    print("装入参数后，模型里的 [0, 0]", model.position_embedding.weight[0, 0].item())
    print("整张位置表完全相同", torch.equal(model.position_embedding.weight, saved_weights["position_embedding.weight"]))


if __name__ == "__main__":
    main()
