import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def main() -> None:
    # 1. 先准备一份完整数据，共 100 个样本。
    inputs = torch.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    targets = -2 * inputs + 3

    # 先配对，保证输入和答案始终一起移动。
    dataset = TensorDataset(inputs, targets)

    # 2. 一次性划分成三份。
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [60, 20, 20],)

    print("全部样本数：", len(dataset))
    print("训练样本数：", len(train_dataset))
    print("验证样本数：", len(validation_dataset))
    print("测试样本数：", len(test_dataset))

    # indices 记录这一份数据在原始 dataset 中的样本编号。
    print("\n训练集前 5 个编号：", train_dataset.indices[:5])
    print("验证集前 5 个编号：", validation_dataset.indices[:5])
    print("测试集前 5 个编号：", test_dataset.indices[:5])

    # 3. 分别设置读取方式。
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    print("\n训练集批次数：", len(train_loader))
    print("验证集批次数：", len(validation_loader))
    print("测试集批次数：", len(test_loader))

    # 4. 看看训练数据实际怎样分批。
    for batch_index, (batch_inputs, batch_targets) in enumerate(
        train_loader,
        start=1,
    ):
        print(
            f"训练第 {batch_index} 批 | "
            f"输入形状 {batch_inputs.shape} | "
            f"答案形状 {batch_targets.shape}"
        )


if __name__ == "__main__":
    main()