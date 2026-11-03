import torch
from torch.utils.data import DataLoader, TensorDataset

from char_vocab import (
    encode,
    char_to_id,
    train_text,
    validation_text,
    test_text,
)

def make_windows(ids, sequence_length):
    samples = []

    for start in range(len(ids) - sequence_length):
        # 1. 从 start 开始，取 sequence_length + 1 个编号。
        window = ids[start:start + sequence_length + 1]

        # 2. 分别取输入和目标。
        inputs = window[:-1]
        targets = window[1:]

        # 3. 把这一对样本加入列表。
        samples.append((inputs, targets))

    return samples

def make_loader(samples, batch_size, shuffle):
    # 没有完整窗口时，明确提示，不继续构建加载器。
    if not samples:
        raise ValueError("没有可用窗口：文本长度至少需要 T + 1 个编号")

    # 1. 分别收集输入和目标。
    all_inputs = []
    all_targets = []

    for inputs, targets in samples:
        all_inputs.append(inputs)
        all_targets.append(targets)

    # 2. 转成整数张量。
    inputs_tensor = torch.tensor(all_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)

    # 3. 按行配对。
    dataset = TensorDataset(inputs_tensor, targets_tensor)

    # 4. 返回可以逐批取数据的加载器。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

train_ids = encode(train_text, char_to_id)
validation_ids = encode(validation_text, char_to_id)
test_ids = encode(test_text, char_to_id)

train_samples = make_windows(train_ids, sequence_length=3)
validation_samples = make_windows(validation_ids, sequence_length=3)
test_samples = make_windows(test_ids, sequence_length=3)

train_loader = make_loader(train_samples, batch_size=2, shuffle=True)
validation_loader = make_loader(validation_samples, batch_size=2, shuffle=False)
test_loader = make_loader(test_samples, batch_size=2, shuffle=False)

for batch_inputs, batch_targets in validation_loader:
    print("验证输入：", batch_inputs)
    print("验证目标：", batch_targets)
    print("验证输入 shape：", batch_inputs.shape)