# LLM Inference Lab

持续学习项目：独立训练 → Transformer/KV Cache → GPU 小实验 → 单卡推理评测 → 一个服务诊断专题。

当前状态：W6 学习骨架，尚无训练/推理实现、测试结果或性能结论。
总控：[ai-infra-roadmap](https://github.com/Zhihui-Cui/ai-infra-roadmap)。
完整阶段计划见 [学习任务](docs/learning-plan.md)。

## 本周从这里开始

进入 learning/01_training，先读任务说明。自己创建 tensor_autograd.py，核对一个简单函数的手算梯度和 autograd；然后写 linear_regression.py，跑通小回归的参数更新。
允许查 API，先尝试，再向 AI 请求提示与审查，不直接复制完整训练循环。

## 环境

先使用现有 WSL/Linux 和 CPU。Python/PyTorch 版本、环境创建和安装命令需在实际配置后写入 docs/environment.md；当前尚未固定依赖，也没有可运行的安装/测试命令。不要为了这一周安装推理框架或购买 GPU。

## 后续边界

learning/ 放教学实现；inference/ 接入成熟引擎；benchmarks/ 放负载、采集和报告；tests/ 放正确性及失败验证。后续目录按实际需要建立。
小模型用于理解机制，真实服务评测使用适合设备的公开预训练模型，不混淆两者。
只使用一套推理引擎、一个模型家族、一个深度问题；量化是替换专题，不是额外必做。
所有结果必须来自真实运行，记录引用、复用代码和个人贡献。
