"""M16·5｜读入 TinyDecoder 的 checkpoint，直接给命令行上的前缀续写。"""

import argparse
from pathlib import Path
import sys

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder
from m15_generation.m15_05_empty_prefix import generate
from m16_reload.m16_04_pack_checkpoint import model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="续写的起始文字，例如：我爱")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="最多新增几个编号")
    default_path = Path(__file__).resolve().parent / "artifacts" / "tiny_decoder_checkpoint.pth"
    parser.add_argument("--checkpoint", type=Path, default=default_path)
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("最多新增的编号数必须是非负数")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyDecoder()
    if checkpoint["model_config"] != model_config(model):
        parser.error("文件中的模型配置与本课的 TinyDecoder 不一致")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    vocab = checkpoint["char_to_id"]
    try:
        result = generate(model, args.prefix, args.max_new_tokens, vocab=vocab)
    except ValueError as error:
        parser.error(str(error))
    print("续写结果", result)


if __name__ == "__main__":
    main()
