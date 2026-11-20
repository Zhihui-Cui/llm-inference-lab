# 以前的练习去哪里找

做 M13 时可以按眼前的问题查一份旧代码。想看一个标量权重怎么更新，打开“梯度”；想知道一个位置的正确字为什么只有一列，打开“多分类”；想分清字符表和位置表，打开“位置向量”。

## 训练基础

| 问题 | 从哪里看 | 与后面练习的关系 |
| --- | --- | --- |
| 梯度怎么算，下一次反向传播前为什么要清零？ | [一个参数的反向传播与手动更新](01_training/01_tensors/gradients_and_manual_update.py) | M13 把同样的梯度交给 SGD 来更新权重。 |
| 一个输入训练出预测值后，怎样比较预测和目标？ | [从 Python 手算 MSE 开始](01_training/01_tensors/mse_in_python.py) · [张量形状](01_training/01_tensors/tensor_shapes.py) · [广播对照](01_training/01_tensors/check_broadcasting.py) | 检查每个预测配的是哪个目标。 |
| 怎样把参数交给优化器，再用新参数前向？ | [线性回归与 SGD](01_training/02_linear_regression/train_with_sgd.py) | 用线性层看训练循环的顺序。 |
| 正确类别怎样选中一行分数里的某一列？ | [多分类交叉熵](01_training/05_classification/multiclass_cross_entropy.py) | “一个包裹的正确类别”换成“一个位置的正确字”，就是 M12 的一道题。 |
| 怎样从类别分数走到训练和检查结果？ | [多分类训练](01_training/05_classification/train_multiclass.py) · [加载多分类参数预测](01_training/05_classification/predict_multiclass.py) · [混淆矩阵](01_training/05_classification/confusion_matrix.py) | 二分类的阈值与误报、漏报另见[二分类训练](01_training/05_classification/train_binary.py)和[二分类指标](01_training/05_classification/binary_metrics.py)。 |
| 数据如何分批，验证时如何评估？ | [批次与 DataLoader](01_training/03_data_and_validation/train_in_batches.py) · [训练集统计量与标准化](01_training/03_data_and_validation/train_standardized_regression.py) | 到 M14 在独立文本上验证时再回看。 |

已有综合练习把多个步骤放在同一个脚本里。双特征回归串起数据划分、训练、验证选模型和重载；可以按需追踪一个样本从输入到预测的过程。[双特征回归综合练习](01_training/06_integrated_practice/m02_two_feature_assignment.py) · [线性回归综合复测](01_training/06_integrated_practice/m01_train_and_reload.py) · [训练阶段原有详细索引](01_training/README.md)

## 从文字到 decoder 的输入

| 问题 | 打开哪份练习 | 对照什么 |
| --- | --- | --- |
| 一个字怎样变成整数编号？ | [字符词表](02_text/char_vocab.py) | `encode` 按训练词表查字。后面的练习用相同的编号。 |
| 同一句话怎样得到输入和下一个字的目标？ | [文本滑动窗口](02_text/m03_text/m04_sequence_windows.py) | `make_windows` 从同一个窗口切出输入和目标，两边相差一个位置。 |
| 一个位置可以读哪些位置？ | [单头 Attention](02_text/m05_attention/m05_m06_causal_attention.py) | 改变第 3 个位置，比较加掩码前后前两个位置的输出。 |
| Q、K、V 是怎么得到的？ | [可训练的投影](02_text/m05_attention/m07_trainable_qkv.py) · [从真实文字开始算 Attention](02_text/m05_attention/text_to_attention.py) | 前者比较更新前后的 Q 投影权重，后者从“我爱学”的字符编号出发。 |
| 位置向量从哪张表来？ | [字符表和位置表](02_text/m05_attention/m10_token_position_embedding.py) | `position_ids` 取 0、1、2。把第三个字换成“我”后，仍然查位置表的第 2 号行。 |
| 每个头各看哪些位置？ | [多头 Attention](02_text/m05_attention/m08_m09_multihead_attention.py) | `split_heads` 把特征列分给各头，每个头都保留整条序列的位置。 |
