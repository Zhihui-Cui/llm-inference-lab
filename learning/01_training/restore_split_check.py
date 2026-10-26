import json
from pathlib import Path

import torch
from torch.utils.data import Subset, TensorDataset


def main() -> None:
    directory = Path(__file__).resolve().parent

    # 1. 读取上一次实验保存的信息。
    config = json.loads(
        (directory / "linear_config.json").read_text(encoding="utf-8")
    )
    indices = torch.load(
        directory / "linear_split_indices.pth",
        weights_only=True,
        map_location="cpu",
    )

    # 2. 按原配置重新生成内容、顺序相同的数据。
    inputs = torch.linspace(
        config["input_min"],
        config["input_max"],
        config["num_samples"],
    ).reshape(-1, 1)

    targets = config["target_slope"] * inputs + config["target_bias"]
    dataset = TensorDataset(inputs, targets)

    # 3. 用保存的索引恢复训练集。
    train_dataset = Subset(dataset, indices["train"])
    validation_dataset = Subset(dataset, indices["validation"])
    test_dataset = Subset(dataset, indices["test"])
    
    print("三个子集的样本数：",
      len(train_dataset),
      len(validation_dataset),
      len(test_dataset))

    print("训练集前 5 个原始索引：", train_dataset.indices[:5])
    print("训练集第一个样本：", train_dataset[0])

if __name__ == "__main__":
    main()