# W07 归档核验：训练脚本隔离运行

- 学习周：W07（2026-10-18～2026-10-24）
- 记录类型：归档核验（verification）
- 截至日期：2026-10-19
- 记录日期：2026-10-19

执行者：Codex，目的是检查本次归档源码；不是学习者独立复测。未改训练逻辑，在隔离源码副本运行，未覆盖主项目已有权重。环境：WSL Ubuntu 24.04、Python 3.12.3、PyTorch 2.14.1+cpu、CPU。

19 个 Python 文本文件（含无后缀 mlp_best_model）均通过 AST 语法解析；实际运行以下 6 个脚本，退出码均为 0。

| 命令（仓库根目录） | 本次结果 / 输出 |
|---|---|
| python -B learning/01_training/mse_practice.py | 0、5/3、9；不等长与空输入均捕获预期 ValueError；[输出](evidence/2026-10-19/mse_practice.txt) |
| python -B learning/01_training/tensor_autograd.py | 梯度 8、累加 14.4、清零后参数 2.2；[输出](evidence/2026-10-19/tensor_autograd.txt) |
| python -B learning/01_training/linear_regression_nn.py | 预测误差断言与重载 allclose 断言通过；[输出](evidence/2026-10-19/linear_regression_nn.txt) |
| python -B learning/01_training/predict_linear.py | 用上一脚本生成的参数预测 [4,1,5]；[输出](evidence/2026-10-19/predict_linear.txt) |
| python -B learning/01_training/linear_independent_check.py | 2,000 次更新；w=1.5、b=-0.7；验证/测试 MSE 均 2.273737e-13；最大绝对误差 4.768372e-7；全部现有断言通过；[输出](evidence/2026-10-19/linear_independent_check.txt) |
| python -B learning/01_training/mlp_square.py | 训练 MSE 7.155484e-3；验证 MSE 1.396699e-2；x=1.5 预测 2.273921；单点误差检查 True；[输出](evidence/2026-10-19/mlp_square.txt) |

[核验清单与源码 SHA-256](evidence/2026-10-19/manifest.json)用于对应实际运行的源文件。输出中仅将隔离副本的绝对目录替换为 <isolated-checkout>；指标未改写。

linear_regression_nn.py 与 linear_independent_check.py 未固定种子，本次结果和本人历史结果的小差异不代表回归；mlp_square.py 固定种子 42，本次复现了其历史输出。其余脚本仅核对源码与语法，不标成本次全量运行通过。

权重、虚拟环境、缓存不纳入提交。历史输出摘要与本次新日志分别存放；不为历史实验补造命令输出、种子或用时。
