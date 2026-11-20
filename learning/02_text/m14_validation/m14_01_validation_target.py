"""M14·1｜同样看到“我爱”，训练文本和验证文本各用哪个字评分？"""

from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id, encode, train_text, validation_text


def main():
    print("训练原文", train_text)
    print("验证原文", validation_text)
    print("未知字编号", char_to_id["<unk>"])

    # 两段原文都以“我爱”开头，下标 2 是紧随其后的那个字。
    train_ids = encode(train_text, char_to_id)
    print("训练：我爱 →", train_text[2], "目标编号", train_ids[2])

    validation_ids = encode(validation_text, char_to_id)
    print("验证：我爱 →", validation_text[2], "目标编号", validation_ids[2])


if __name__ == "__main__":
    main()
