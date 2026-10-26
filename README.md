# LLM Inference Lab

持续学习项目：独立训练 → Transformer/KV Cache → GPU 小实验 → 单卡推理评测 → 一个服务诊断专题。

更新：2026-10-26，当前 W08（10/25–10/31）。CPU 上已跑通 Python MSE、Tensor/autograd、线性回归、MLP、分批训练、验证/测试和参数保存加载。当前属于**本人练习 + AI 提示、审查及参考代码辅助完成**；G1 独立训练门槛仍待验收，尚未开始 Transformer 或推理服务评测。

[总控与本周任务](https://github.com/Zhihui-Cui/ai-infra-roadmap/blob/main/now.md) · [阶段计划](docs/learning-plan.md) · [练习索引](learning/01_training/README.md)

## 最新学习记录

| 截至日期 | 学习周 | 类型 | 文档 |
|---|---|---|---|
| 2026-10-25 | W08 | 阶段记录 | [M01 实验追溯与重载检查（10/26 补记）](reports/2026-10-25-W08-checkpoint.md) |
| 2026-10-24 | W07 | 周复盘 | [分类训练、评价指标与保存加载](reports/2026-10-24-W07-review.md) |
| 2026-10-23 | W07 | 阶段记录 | [二分类指标与阈值选择](reports/2026-10-23-W07-checkpoint.md) |
| 2026-10-21 | W07 | 阶段记录 | [训练可靠性与二分类](reports/2026-10-21-W07-checkpoint.md) |
| 2026-10-19 | W07 | 阶段记录 | [数据划分与完整训练流程](reports/2026-10-19-W07-checkpoint.md) |
| 2026-10-19 | W07 | 归档核验 | [训练脚本隔离运行](reports/2026-10-19-W07-verification.md) |
| 2026-10-17 | W06 | 周复盘 | [Python/PyTorch 与最小训练](reports/2026-10-17-W06-review.md) |

[全部学习记录](reports/README.md) · [记录模板](reports/TEMPLATE.md) · [命名与记录规范](https://github.com/Zhihui-Cui/ai-infra-roadmap/blob/main/docs/documentation-conventions.md)

## 环境与运行

使用 WSL Ubuntu 24.04、Python 3.12.3、PyTorch 2.14.1+cpu。详细安装、迁移和验证过程见 [环境记录](docs/environment.md)，依赖见 [requirements-cpu.txt](requirements-cpu.txt)。

在仓库根目录、已创建并激活的 Python 3.12 环境内执行：

```bash
python -m pip install -r requirements-cpu.txt
python learning/01_training/mse_practice.py
python learning/01_training/linear_independent_check.py
python learning/01_training/mlp_square.py
```

最后两个命令分别展示完整线性训练链和非线性拟合。文件名中的 independent 只表示练习题名，不代表已无参考独立完成。部分脚本未固定随机种子，精确指标会变化。

参数文件由训练脚本在本地生成，已由忽略规则排除；运行仅预测脚本前需先运行相应训练脚本，见练习索引。

## 接下来

1. 两次指导下故障闭环、标准化和二分类入门已完成，不再作为未学任务重复安排。
2. 类别不平衡、Precision/Recall 统计、无报警零分母案例、按代价选择阈值及独立恢复阈值已完成指导练习（见 10/23 阶段记录，依据学习者提供的运行输出）。10/24 已完成三分类及混淆矩阵指导练习；10/25 回顾已能正确说明指标分母，并本人确认保存加载打印结果一致。详细证据与验证边界见 W07 周复盘。
3. M01 已完成配置/种子/划分索引保存与读回一致、线性回归全精度重载及最终测试，依据本人 10/25 运行输出；助手提供完整修订参考。尚待从保存信息重建划分并检查隔离/覆盖，以及归档真实最终源码，M01 暂不整项勾选。详见 W08 阶段记录；远程训练脚本尚未包含此次本地修改。
4. M01 收尾后进入 M02 综合独立变体（G1），允许查 API，按任务依赖推进。已有训练和重载检查不重做；G1 独立验收仍待完成。

任务入口：[W06 #1](https://github.com/Zhihui-Cui/llm-inference-lab/issues/1) · [W07 #2](https://github.com/Zhihui-Cui/llm-inference-lab/issues/2) · [W08 / G1 #3](https://github.com/Zhihui-Cui/llm-inference-lab/issues/3)。

## 仓库边界

learning/ 放教学练习；inference/ 接入成熟引擎；benchmarks/ 放负载、采集和报告；tests/ 放正确性及失败验证，按阶段需要建立。小模型练习不等于真实服务性能评测。

只使用一套推理引擎、一个模型家族、一个深入问题；量化作为替换专题，不额外叠加。不提交虚拟环境、权重、完整数据集或密钥。真实运行结果、参考代码和个人贡献分别记录。
