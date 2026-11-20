# 训练基础练习

按问题找代码。每个目录对应一个主题，从短实验逐步走到完整训练。需要回忆 M13 的梯度和更新时，先看[反向传播与手动更新](01_tensors/gradients_and_manual_update.py)；想回忆正确字的损失，查看[多分类交叉熵](05_classification/multiclass_cross_entropy.py)。

| 学习顺序 | 什么时候回来查 |
| --- | --- |
| [1. 张量与梯度](01_tensors/) | 怎样读形状、算损失、解释一次参数更新？ |
| [2. 线性回归](02_linear_regression/) | 怎样把手写权重换成 Linear 和 SGD？ |
| [3. 数据与验证](03_data_and_validation/) | 怎样分批、选模型、标准化并检查误差？ |
| [4. 非线性回归](04_nonlinear_regression/) | MLP 怎样拟合平方函数、保存最佳模型？ |
| [5. 分类](05_classification/) | 怎样从分数得到损失、类别和评价指标？ |
| [6. 综合练习](06_integrated_practice/) | 怎样独立串起训练、保存、划分恢复和复核？ |

## 1. 张量与梯度

- [用 Python 算均方误差](01_tensors/mse_in_python.py) · `mse_in_python.py`
- [用张量计算相同的 MSE](01_tensors/mse_with_tensors.py) · `mse_with_tensors.py`
- [看行、列和标量的形状](01_tensors/tensor_shapes.py) · `tensor_shapes.py`
- [定位意外广播](01_tensors/check_broadcasting.py) · `check_broadcasting.py`
- [反向传播、累积梯度与手动更新](01_tensors/gradients_and_manual_update.py) · `gradients_and_manual_update.py`
- [矩阵乘法如何给每个样本产生预测](01_tensors/matrix_prediction.py) · `matrix_prediction.py`

## 2. 线性回归

- [手写单特征回归](02_linear_regression/fit_one_feature.py) · `fit_one_feature.py`
- [换成负斜率重新拟合](02_linear_regression/fit_negative_slope.py) · `fit_negative_slope.py`
- [两个特征怎样共同预测](02_linear_regression/fit_two_features.py) · `fit_two_features.py`
- [查看 Linear 的参数与输出](02_linear_regression/inspect_linear_layer.py) · `inspect_linear_layer.py`
- [用 Linear 和 SGD 训练、保存](02_linear_regression/train_with_sgd.py) · `train_with_sgd.py`
- [加载线性层权重做预测](02_linear_regression/predict_from_weights.py) · `predict_from_weights.py`

## 3. 数据与验证

- [用 DataLoader 分批训练](03_data_and_validation/train_in_batches.py) · `train_in_batches.py`
- [划分训练、验证、测试样本](03_data_and_validation/split_dataset.py) · `split_dataset.py`
- [按验证误差选权重并早停](03_data_and_validation/select_best_regression.py) · `select_best_regression.py`
- [加载所选权重，在固定输入上检验](03_data_and_validation/evaluate_saved_regression.py) · `evaluate_saved_regression.py`
- [标签有噪声时观察误差](03_data_and_validation/fit_noisy_targets.py) · `fit_noisy_targets.py`
- [用训练集统计量标准化输入](03_data_and_validation/standardize_inputs.py) · `standardize_inputs.py`
- [串起标准化、训练、保存和重载](03_data_and_validation/train_standardized_regression.py) · `train_standardized_regression.py`

## 4. 非线性回归

- [用两层网络拟合平方函数](04_nonlinear_regression/fit_square_with_mlp.py) · `fit_square_with_mlp.py`
- [按验证集选 MLP 并重载](04_nonlinear_regression/select_best_mlp.py) · `select_best_mlp.py`

## 5. 分类

- [阈值、误报漏报与二分类指标](05_classification/binary_metrics.py) · `binary_metrics.py`
- [训练二分类模型并选择阈值](05_classification/train_binary.py) · `train_binary.py`
- [重载二分类模型和阈值](05_classification/predict_binary.py) · `predict_binary.py`
- [用正确类别列计算交叉熵](05_classification/multiclass_cross_entropy.py) · `multiclass_cross_entropy.py`
- [训练并保存多分类模型](05_classification/train_multiclass.py) · `train_multiclass.py`
- [重载多分类模型做预测](05_classification/predict_multiclass.py) · `predict_multiclass.py`
- [从预测与目标填写混淆矩阵](05_classification/confusion_matrix.py) · `confusion_matrix.py`

## 6. 综合练习

- [M01 · 训练、保存配置、重载与核对](06_integrated_practice/m01_train_and_reload.py) · `m01_train_and_reload.py`
- [M01 · 从保存的索引恢复划分](06_integrated_practice/m01_restore_split.py) · `m01_restore_split.py`
- [M02 · 评估、选优与加载的函数练习](06_integrated_practice/m02_checkpoint_functions.py) · `m02_checkpoint_functions.py`
- [G1 · 两个输入特征的综合训练练习](06_integrated_practice/m02_two_feature_assignment.py) · `m02_two_feature_assignment.py`

M02 函数练习用于复习局部操作，综合训练可独立运行。课程进度与独立掌握程度见当时的学习记录。

## 训练和重载怎样配对

| 想检查什么 | 第一次生成文件 | 再运行 |
| --- | --- | --- |
| 线性层预测 | [train_with_sgd.py](02_linear_regression/train_with_sgd.py) | [predict_from_weights.py](02_linear_regression/predict_from_weights.py) |
| 验证集选出的线性权重 | [select_best_regression.py](03_data_and_validation/select_best_regression.py) | [evaluate_saved_regression.py](03_data_and_validation/evaluate_saved_regression.py) |
| 二分类的权重与阈值 | [train_binary.py](05_classification/train_binary.py) | [predict_binary.py](05_classification/predict_binary.py) |
| 多分类的权重 | [train_multiclass.py](05_classification/train_multiclass.py) | [predict_multiclass.py](05_classification/predict_multiclass.py) |
| M01 的划分和配置 | [m01_train_and_reload.py](06_integrated_practice/m01_train_and_reload.py) | [m01_restore_split.py](06_integrated_practice/m01_restore_split.py) |

从 WSL 仓库根目录运行，例如：

```bash
.venv/bin/python learning/01_training/01_tensors/gradients_and_manual_update.py
```

权重、运行配置和划分索引按主题保存在各自的 `artifacts/`。训练入口会写本次实验结果；仅重载时可直接用已有文件。两份标准化练习现在各写一份权重，迁移前已有的 `linear_standardized_checkpoint.pth` 保留在数据与验证主题的 `artifacts/` 中。

[W06 学习记录](../../reports/2026-10-17-W06-review.md) · [W07 阶段记录](../../reports/2026-10-19-W07-checkpoint.md) · [训练脚本核验](../../reports/2026-10-19-W07-verification.md) · [标准化与二分类记录](../../reports/2026-10-21-W07-checkpoint.md)

<details>
<summary>用以前笔记中的文件名查找新位置</summary>

| 原文件名 | 现在的位置 |
| --- | --- |
| `binary_classification.py` | [05_classification/train_binary.py](05_classification/train_binary.py) |
| `broadcast_debug.py` | [01_tensors/check_broadcasting.py](01_tensors/check_broadcasting.py) |
| `classification_metrics.py` | [05_classification/binary_metrics.py](05_classification/binary_metrics.py) |
| `confusion_matrix_practice.py` | [05_classification/confusion_matrix.py](05_classification/confusion_matrix.py) |
| `dataloader_practice.py` | [03_data_and_validation/train_in_batches.py](03_data_and_validation/train_in_batches.py) |
| `dataset_split.py` | [03_data_and_validation/split_dataset.py](03_data_and_validation/split_dataset.py) |
| `g1_two_feature_training.py` | [06_integrated_practice/m02_two_feature_assignment.py](06_integrated_practice/m02_two_feature_assignment.py) |
| `linear_independent_check.py` | [06_integrated_practice/m01_train_and_reload.py](06_integrated_practice/m01_train_and_reload.py) |
| `linear_noisy.py` | [03_data_and_validation/fit_noisy_targets.py](03_data_and_validation/fit_noisy_targets.py) |
| `linear_regression.py` | [02_linear_regression/fit_one_feature.py](02_linear_regression/fit_one_feature.py) |
| `linear_regression_independent.py` | [02_linear_regression/fit_negative_slope.py](02_linear_regression/fit_negative_slope.py) |
| `linear_regression_nn.py` | [02_linear_regression/train_with_sgd.py](02_linear_regression/train_with_sgd.py) |
| `linear_regression_two_features.py` | [02_linear_regression/fit_two_features.py](02_linear_regression/fit_two_features.py) |
| `linear_review.py` | [03_data_and_validation/select_best_regression.py](03_data_and_validation/select_best_regression.py) |
| `linear_standardized.py` | [03_data_and_validation/train_standardized_regression.py](03_data_and_validation/train_standardized_regression.py) |
| `m02_part_practice.py` | [06_integrated_practice/m02_checkpoint_functions.py](06_integrated_practice/m02_checkpoint_functions.py) |
| `matrix_prediction.py` | [01_tensors/matrix_prediction.py](01_tensors/matrix_prediction.py) |
| `mlp_best_model` | [04_nonlinear_regression/select_best_mlp.py](04_nonlinear_regression/select_best_mlp.py) |
| `mlp_square.py` | [04_nonlinear_regression/fit_square_with_mlp.py](04_nonlinear_regression/fit_square_with_mlp.py) |
| `mse_practice.py` | [01_tensors/mse_in_python.py](01_tensors/mse_in_python.py) |
| `multiclass_practice.py` | [05_classification/multiclass_cross_entropy.py](05_classification/multiclass_cross_entropy.py) |
| `multiclass_training.py` | [05_classification/train_multiclass.py](05_classification/train_multiclass.py) |
| `nn_linear_practice.py` | [02_linear_regression/inspect_linear_layer.py](02_linear_regression/inspect_linear_layer.py) |
| `predict_binary.py` | [05_classification/predict_binary.py](05_classification/predict_binary.py) |
| `predict_linear.py` | [02_linear_regression/predict_from_weights.py](02_linear_regression/predict_from_weights.py) |
| `predict_multiclass.py` | [05_classification/predict_multiclass.py](05_classification/predict_multiclass.py) |
| `restore_split_check.py` | [06_integrated_practice/m01_restore_split.py](06_integrated_practice/m01_restore_split.py) |
| `standardization_practice.py` | [03_data_and_validation/standardize_inputs.py](03_data_and_validation/standardize_inputs.py) |
| `tensor_autograd.py` | [01_tensors/gradients_and_manual_update.py](01_tensors/gradients_and_manual_update.py) |
| `tensor_mse.py` | [01_tensors/mse_with_tensors.py](01_tensors/mse_with_tensors.py) |
| `tensor_shapes.py` | [01_tensors/tensor_shapes.py](01_tensors/tensor_shapes.py) |
| `test_linear_review.py` | [03_data_and_validation/evaluate_saved_regression.py](03_data_and_validation/evaluate_saved_regression.py) |

</details>
