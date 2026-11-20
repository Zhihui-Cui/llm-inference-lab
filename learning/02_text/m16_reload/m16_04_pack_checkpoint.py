"""M16·4｜把已验证的权重、当前模型的配置和训练词表存进同一个 checkpoint。"""

from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder
from char_vocab import char_to_id, encode


def model_config(model):
    return {
        "architecture": "TinyDecoder",
        "vocab_size": model.token_embedding.num_embeddings,
        "d_model": model.token_embedding.embedding_dim,
        "num_heads": model.block.attention.num_heads,
        "d_ff": model.block.ffn[0].out_features,
        "max_positions": model.position_embedding.num_embeddings,
        "dropout": model.block.attention.dropout,
    }


def main():
    folder = Path(__file__).resolve().parent / "artifacts"
    weights = torch.load(folder / "m14_four_epochs.pth", map_location="cpu", weights_only=True)
    model = TinyDecoder()
    model.load_state_dict(weights)
    config = model_config(model)
    checkpoint = {"model_state_dict": weights, "model_config": config}
    checkpoint["char_to_id"] = char_to_id

    checkpoint_path = folder / "tiny_decoder_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print("文件中的三项", list(restored))
    print("文件中的模型配置", restored["model_config"])
    print("用文件中的词表编码“我爱”", encode("我爱", restored["char_to_id"]))


if __name__ == "__main__":
    main()
