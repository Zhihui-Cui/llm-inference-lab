"""M14·2｜只从验证文本取窗口，看看一共要给哪些位置评分。"""

from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id, encode, validation_text
from m03_text.m04_sequence_windows import make_windows


def main():
    validation_ids = encode(validation_text, char_to_id)
    print("验证原文", validation_text)
    print("验证编号", validation_ids)

    validation_windows = make_windows(validation_ids, 3)
    print(validation_windows)


if __name__ == "__main__":
    main()
