import torch

# 类别编号：0 轻件、1 中件、2 重件。
targets = torch.tensor([0, 0, 1, 1, 2, 2])
predictions = torch.tensor([0, 1, 1, 2, 2, 2])

# 创建 3 行 3 列的整数表，初始全部为 0。
matrix = torch.zeros((3, 3), dtype=torch.long)

# 每次取出同一个样本的真实类别和预测类别。
for target, prediction in zip(targets, predictions):
    true_id = target.item()
    predicted_id = prediction.item()

    # 行是真实类别，列是预测类别。
    matrix[true_id, predicted_id] += 1

print("混淆矩阵（行：真实，列：预测）：")
print(matrix)

# 每一行相加，得到各类实际数量。
actual_counts = matrix.sum(dim=1)

# 每一列相加，得到各类预测数量。
predicted_counts = matrix.sum(dim=0)

# 对角线是各类预测正确的数量。
correct_counts = matrix.diagonal()

print("各类实际数量：", actual_counts)
print("各类预测数量：", predicted_counts)
print("各类正确数量：", correct_counts)