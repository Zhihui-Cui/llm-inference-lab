# CPU 学习环境

环境配置与基础验证已完成；这不代表 Tensor、autograd 或训练任务已通过独立验收。

## 实际环境

- 安装开始：2026-10-12；运行验证：2026-10-13（Asia/Shanghai）。
- 验证时当前提交：`62cb54db005a39c608e0144a2bd8d952893f8037`（不含未提交的练习与环境记录）。
- 操作系统：WSL2 / Ubuntu 24.04；`Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`。
- Python：`3.12.3`。
- 项目解释器：`/home/zhihui/projects/llm-inference-lab/.venv/bin/python`。
- PyTorch：`2.14.1+cpu`；NumPy：`2.5.2`。
- pip：`pip 26.2.1 from /home/zhihui/projects/llm-inference-lab/.venv/lib/python3.12/site-packages/pip (python 3.12)`。
- CPU：12th Gen Intel(R) Core(TM) i5-1240P。
- 运算设备：CPU；PyTorch CUDA 构建：`None`。本次 GPU、驱动与 CUDA 不适用，未据此判断机器是否有 GPU。
- 依赖清单：根目录 `requirements-cpu.txt`，来源为安装完成后的 `pip freeze`。

## 在 VS Code 中使用

1. 按 `Ctrl+Shift+P`，连接 Ubuntu-24.04，并打开 `/home/zhihui/projects/llm-inference-lab`。
2. 左下角确认显示 WSL。若提示，给 WSL 端安装 Microsoft Python 扩展。
3. 执行 `Python: Select Interpreter`，选择项目 `.venv/bin/python`。
4. 新开 WSL 终端，在项目根目录激活：

```bash
cd /home/zhihui/projects/llm-inference-lab
source .venv/bin/activate
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
python learning/01_training/mse_practice.py
```

终端中 Python 路径应落在本项目 `.venv/bin/` 内。该 `.venv` 是 Linux 环境，不能在 Windows PowerShell 中直接当成 Windows Python 使用。退出环境用 `deactivate`。

## 原环境创建和安装记录（迁移时重建）

当前 Ubuntu 有 Python 3.12，但没有 pip 和 ensurepip，因此创建环境时使用：

```bash
python3 -m venv --without-pip .venv
```

pip 使用官方 `https://bootstrap.pypa.io/get-pip.py` 引导脚本安装。脚本下载在 Codex 工作目录，实际执行的环境内安装命令为：

```bash
.venv/bin/python /mnt/c/Users/28505/Documents/Codex/2026-10-12/10-10-11-10-17-python/work/get-pip.py --no-cache-dir --index-url https://pypi.org/simple
.venv/bin/python -m pip install --no-cache-dir --only-binary=:all: torch numpy --index-url https://download.pytorch.org/whl/cpu --timeout 30 --retries 2
```

复现时先准备一个带 pip 的 Python 3.12 虚拟环境，再安装已记录版本：

```bash
python -m pip install -r requirements-cpu.txt
python -m pip check
```

`.venv/` 已在 `.gitignore` 中，依赖清单和环境文档可纳入版本管理。

## 已执行的验证

- 依赖检查：pip check 输出 No broken requirements found.。

- PyTorch 导入成功，版本为 `2.14.1+cpu`。
- 在 CPU 上创建长度为 3 的 Tensor，shape 为 `[3]`。
- 对 `[1, 2, 3]` 的平方和求导：输出 `14.0`，梯度为 `[2.0, 4.0, 6.0]`；断言通过。
- NumPy 数组转 Tensor 并求和，断言通过。
- 现有纯 Python `mse` 对 `[1, 2, 3]` 和 `[2, 2, 5]` 返回 `5 / 3`，断言通过。

上述检查仅验收环境；未提供 Tensor MSE 练习的完整答案。

## 问题与处理

- 原 PowerShell 中的 `python` 指向 `C:\msys64\ucrt64\bin\python.exe`，不是项目专用环境，导入 torch 报 `ModuleNotFoundError`。
- 按项目约定使用现有 WSL/Linux 和 CPU，依赖安装在项目 `.venv` 内。
- WSL 启动有 localhost 代理提示；本次下载与验证结果以实际完成记录为准。
- AI 协助范围：环境检查、创建、依赖安装、基础运行验证与本文记录。学习者仍需自行完成 Tensor 运算和训练练习。

参考：[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)、[pip 官方安装说明](https://pip.pypa.io/en/stable/installation/)、[VS Code WSL](https://code.visualstudio.com/docs/remote/wsl)。

## 2026-10-13：迁移到 WSL 原生文件系统

- 当前主项目：`/home/zhihui/projects/llm-inference-lab`。
- E 盘原项目保留作为迁移备份；以后在上述 Linux 目录学习，避免两处交替修改。
- Git 历史及未提交文件已复制；复制时 43 个文件逐一校验一致。
- 虚拟环境在目标路径重新创建，没有直接搬用旧 `.venv`。
- 使用原有 `requirements-cpu.txt` 固定版本安装，pip freeze 集合对照一致。
- pip check、CPU Tensor、autograd、NumPy 互操作及两份现有练习运行通过。
- AI 工作：环境迁移与验证，未改写学习者的两份练习代码。

实际重建命令（在新项目目录）：

```bash
python3 -m venv --without-pip .venv
.venv/bin/python /mnt/c/Users/28505/Documents/Codex/2026-10-12/10-10-11-10-17-python/work/get-pip.py --no-cache-dir --index-url https://pypi.org/simple
.venv/bin/python -m pip install --no-cache-dir --only-binary=:all: -r requirements-cpu.txt --timeout 30 --retries 2
```

后续在 WSL 终端使用：

```bash
cd /home/zhihui/projects/llm-inference-lab
code .
source .venv/bin/activate
python learning/01_training/tensor_mse.py
```

VS Code 使用 WSL 窗口，并选择 `/home/zhihui/projects/llm-inference-lab/.venv/bin/python`。

### 导入时间观察

旧环境先前单次观测：import torch 为 44.945 秒。
新环境以下两次使用独立 Python 进程，均带 `-B`，与此前诊断一致。
测量区间是 Python 进程内部的 torch 导入，不含 WSL 启动和工具调度时间。
这不是严格冷缓存对照实验，不能将比值当作通用加速倍数。

第 1 次：

```text
Python started; importing torch...
torch import: 1.868 s
first MSE including tensor creation: 0.000831 s
1000 repeated MSE calculations: 0.006953 s
torch location: /home/zhihui/projects/llm-inference-lab/.venv/lib/python3.12/site-packages/torch/__init__.py
```

第 2 次：

```text
Python started; importing torch...
torch import: 1.444 s
first MSE including tensor creation: 0.000867 s
1000 repeated MSE calculations: 0.006616 s
torch location: /home/zhihui/projects/llm-inference-lab/.venv/lib/python3.12/site-packages/torch/__init__.py
```

依据：[微软 WSL 文件系统建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)。
