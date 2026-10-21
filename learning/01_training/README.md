# 训练阶段练习索引

截至 2026-10-21。保存学习过程中的各阶段版本，未统一重构为训练框架。允许查语法/API；在用户明确请求时提供过完整参考实现，因此“运行通过”和“独立验收”分开记录。

## 建议阅读顺序

| 阶段 | 文件 | 核心内容 |
|---|---|---|
| Python MSE | [mse_practice.py](mse_practice.py) | 数值返回、长度/空输入异常、负数案例 |
| Tensor | [tensor_mse.py](tensor_mse.py)、[tensor_shapes.py](tensor_shapes.py) | mean()、标量 shape、广播与 reshape |
| 梯度 | [tensor_autograd.py](tensor_autograd.py) | backward、grad 累加、手动更新、清零 |
| 单特征回归 | [linear_regression.py](linear_regression.py)、[变体](linear_regression_independent.py) | 手写参数更新；变体 y=-3x+2 |
| 矩阵与多特征 | [matrix_prediction.py](matrix_prediction.py)、[双特征训练](linear_regression_two_features.py) | @、shape、两个权重 |
| 模块和优化器 | [nn_linear_practice.py](nn_linear_practice.py)、[linear_regression_nn.py](linear_regression_nn.py) | Linear、SGD、验证、保存加载 |
| 只预测 | [predict_linear.py](predict_linear.py) | 先运行 linear_regression_nn.py 生成参数 |
| 分批训练 | [dataloader_practice.py](dataloader_practice.py) | Dataset、DataLoader、epoch/batch、加权 MSE |
| 非线性拟合 | [mlp_square.py](mlp_square.py) | Linear→ReLU→Linear；训练误差不等于泛化误差 |
| 最佳参数 | [mlp_best_model](mlp_best_model) | 验证集选模型、立即保存、重载核对；原文件无 .py 后缀 |
| 数据划分 | [dataset_split.py](dataset_split.py) | random_split、Subset、末批不足 batch_size |
| 完整线性流程 | [linear_review.py](linear_review.py) | 60/20/20 划分、最佳参数、早停、最终测试 |
| 单独测试 | [test_linear_review.py](test_linear_review.py) | 先运行 linear_review.py；固定输入误差检查 |
| 含噪声实验 | [linear_noisy.py](linear_noisy.py) | 噪声标准差 0.2，误差不应追求到零 |
| 综合复测 | [linear_independent_check.py](linear_independent_check.py) | 120 样本、80/20/20、2,000 次更新、重载和数值断言 |

## 10/21 新增与更新

| 文件 | 内容 |
|---|---|
| [broadcast_debug.py](broadcast_debug.py) | 广播错误的修复前后对照 |
| [standardization_practice.py](standardization_practice.py) | 训练集统计量、标准化、必要配置保存加载 |
| [binary_classification.py](binary_classification.py) | BCE、阈值、误报/漏报、验证网格搜索、教学测试、模型与阈值保存 |
| [predict_binary.py](predict_binary.py) | 先运行 binary_classification，再独立恢复参数与阈值 |
| [linear_independent_check.py](linear_independent_check.py) | 恢复 step 后的调试版本；参数在 step 后打印，梯度来自更新前 |

[本次详细报告和核验日志](../../reports/2026-10-21-W07-checkpoint.md)。标准化和二分类各自独立，二分类没有输入标准化。仅保存推理状态，不包括恢复优化器训练所需的状态。

所有脚本从仓库根目录用 python 加对应路径运行。无后缀的 MLP 文件也是 Python 文本，可用 python learning/01_training/mlp_best_model 运行。保存文件位于脚本旁边；训练脚本可能覆盖自己的旧参数文件。

## 当前证据如何读

- [W06 记录](../../reports/2026-10-17-W06-review.md)保存当时输出摘录；脚本后来可能调整了轮数/数据，不保证最新代码逐字复现每条历史日志。
- [W07 快照](../../reports/2026-10-19-W07-checkpoint.md)分别记录 10/18 和 10/19，不提前记到 W06。
- [本次核验](../../reports/2026-10-19-W07-verification.md)记录 19 个文件的语法检查及 6 个脚本的实际运行；其他脚本不冒充本次已运行。
- 全数据评估按批大小加权；这里每个样本一个输出。train_eval_loader 使用训练集本身，不单独划走一份数据。
- eval() 切换层的行为；no_grad() 控制计算图；backward() 求导；optimizer.step() 更新参数。当前 Linear/ReLU 的 train/eval 前向行为相同。
- 训练 MSE 很小、单点误差通过、重载一致，各自只能证明对应检查；不能互相替代，也不能代替独立编写能力。

部分脚本有随机种子，部分没有；不为历史实验补写不存在的种子。标准化统计量、分类阈值和两个指导下故障闭环已补齐；划分索引持久化、完整复现配置及延迟独立复做仍待完善。
