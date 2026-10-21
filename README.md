# LLM Inference Lab

持续学习项目：独立训练 → Transformer/KV Cache → GPU 小实验 → 单卡推理评测 → 一个服务诊断专题。

更新：2026-10-21，当前 W7（10/18–10/24）。CPU 上已跑通 Python MSE、Tensor/autograd、线性回归、MLP、分批训练、验证/测试和参数保存加载。当前属于**本人练习 + AI 提示、审查及参考代码辅助完成**；G1 独立训练门槛仍待验收，尚未开始 Transformer 或推理服务评测。

[总控与本周任务](https://github.com/Zhihui-Cui/ai-infra-roadmap/blob/main/now.md) · [阶段计划](docs/learning-plan.md) · [练习索引](learning/01_training/README.md)

## 本次学习记录

- [W7：截至 10/21 的进展](reports/2026-10-21-W07-checkpoint.md)：两次故障闭环、训练集标准化、二分类阈值搜索及模型/阈值独立加载；含五个脚本的隔离核验日志。

- [W6：10/11–10/17 学习总结](reports/2026-10-17-W06.md)：从纯 Python 到最小训练，并提前接触部分 W7 内容。
- [W7：截至 10/19 的进展](reports/2026-10-19-W07-checkpoint.md)：数据划分、早停、含噪声实验与完整线性流程复测。
- [10/19 代码归档核验](reports/2026-10-19-verification.md)：独立于历史学习成绩的本次实际运行，附输出及源码摘要。

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
2. 下一小节：类别不平衡；补 Precision/Recall 的统计与零分母处理，多分类尚未学习。
3. 月底一次综合变体检查 G1，允许查 API；完善配置/划分追溯，不用重复抄写完整训练循环。G1 独立验收仍待完成。

任务入口：[W6 #1](https://github.com/Zhihui-Cui/llm-inference-lab/issues/1) · [W7 #2](https://github.com/Zhihui-Cui/llm-inference-lab/issues/2) · [W8 / G1 #3](https://github.com/Zhihui-Cui/llm-inference-lab/issues/3)。

## 仓库边界

learning/ 放教学练习；inference/ 接入成熟引擎；benchmarks/ 放负载、采集和报告；tests/ 放正确性及失败验证，按阶段需要建立。小模型练习不等于真实服务性能评测。

只使用一套推理引擎、一个模型家族、一个深入问题；量化作为替换专题，不额外叠加。不提交虚拟环境、权重、完整数据集或密钥。真实运行结果、参考代码和个人贡献分别记录。
