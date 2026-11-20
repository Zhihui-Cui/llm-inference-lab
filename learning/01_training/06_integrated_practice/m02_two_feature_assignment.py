import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ============================================================
# 1. 根据配置生成数据，并在训练开始前划分一次
# ============================================================
def make_data(config):
    # 独立生成器：控制数据生成和随机划分。
    generator = torch.Generator().manual_seed(config["seed"])

    # torch.rand 先生成 [0, 1) 范围内的随机数，
    # 再缩放到配置指定的输入范围。
    # inputs.shape = [样本数, 2]
    inputs = (
        torch.rand(config["num_samples"], 2, generator=generator)
        * (config["input_max"] - config["input_min"])
        + config["input_min"]
    )

    # 每个样本有两个特征，目标关系为：
    # y = 1.5 * x1 - 2.0 * x2 + 0.7
    #
    # 使用 0:1 和 1:2 保留第二个维度，
    # 使 targets.shape = [样本数, 1]。
    targets = (
        config["target_weights"][0] * inputs[:, 0:1]
        + config["target_weights"][1] * inputs[:, 1:2]
        + config["target_bias"]
    )

    dataset = TensorDataset(inputs, targets)

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        config["split_sizes"],
        generator=generator,
    )

    return train_dataset, validation_dataset, test_dataset


# ============================================================
# 2. 评估函数：计算一整份数据的 MSE，不更新参数
# ============================================================
def evaluate(model, data_loader, loss_fn) -> float:
    model.eval()

    total_squared_error = 0.0
    total_samples = 0

    # 评估时不需要记录用于反向传播的计算图。
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_predictions = model(batch_inputs)

            batch_loss = loss_fn(batch_predictions, batch_targets)
            current_batch_size = batch_inputs.shape[0]

            total_squared_error += batch_loss.item() * current_batch_size
            total_samples += current_batch_size

    assert total_samples > 0

    return total_squared_error / total_samples


def main() -> None:
    # ========================================================
    # 3. 实验配置与文件路径
    # ========================================================
    config = {
        "num_samples": 123,
        "input_min": -3.0,
        "input_max": 3.0,
        "target_weights": [1.5, -2.0],
        "target_bias": 0.7,
        "split_sizes": [83, 20, 20],
        "batch_size": 8,
        "num_epochs": 200,
        "learning_rate": 0.03,
        "log_interval": 20,
        "seed": 42,
    }

    # 文件都保存在当前脚本所在目录。
    directory = Path(__file__).resolve().parent / "artifacts"
    directory.mkdir(exist_ok=True)

    config_path = directory / "g1_two_feature_config.json"
    split_path = directory / "g1_two_feature_split_indices.pth"
    model_path = directory / "g1_two_feature_weights.pth"

    # 控制模型初始化等使用全局生成器的随机操作。
    torch.manual_seed(config["seed"])

    # ========================================================
    # 4. 创建数据，保存配置与实际划分索引
    # ========================================================
    train_dataset, validation_dataset, test_dataset = make_data(
        config
    )

    # indices 保存的是样本在原始数据集中的位置。
    split_indices = {
        "train": train_dataset.indices,
        "validation": validation_dataset.indices,
        "test": test_dataset.indices,
    }

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    torch.save(split_indices, split_path)

    print(
        f"训练样本：{len(train_dataset)} | "
        f"验证样本：{len(validation_dataset)} | "
        f"测试样本：{len(test_dataset)}"
    )

    # ========================================================
    # 5. 创建四个加载器
    # ========================================================

    # 专门给训练加载器一个生成器，控制样本打乱顺序。
    # 这样额外调用评估函数，也不会消耗这个生成器的随机状态。
    train_generator = torch.Generator().manual_seed(config["seed"])

    # 用于参数更新：每轮打乱，保留最后不足一批的样本。
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
        generator=train_generator,
    )

    # 使用同一份训练数据，仅用于评估当前模型的整体训练 MSE。
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # ========================================================
    # 6. 创建模型、优化器和损失函数
    # ========================================================

    # 两个输入特征，一个输出。
    # weight.shape = [1, 2]，bias.shape = [1]。
    model = nn.Linear(2, 1)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
    )

    # 默认返回当前批所有输出元素的平均平方误差。
    loss_fn = nn.MSELoss()

    best_validation_loss = float("inf")
    best_epoch = 0
    update_count = 0

    # 固定输入用于检查参数保存、加载前后的预测一致性。
    # 它们不参与参数更新，也不参与最佳模型的选择。
    check_inputs = torch.tensor(
        [
            [-2.0, 1.0],
            [0.0, 0.0],
            [1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    best_predictions = None

    # ========================================================
    # 7. 训练：每批更新参数，每轮结束后验证
    # ========================================================
    for epoch in range(config["num_epochs"]):
        # 上一轮 evaluate() 会切换到评估模式，
        # 所以每轮训练开始时重新切换到训练模式。
        model.train()

        for batch_inputs, batch_targets in train_loader:
            # ① 清除上一批留下的梯度。
            optimizer.zero_grad()

            # ② 根据当前参数进行预测。
            batch_predictions = model(batch_inputs)

            # ③ 计算损失。
            batch_loss = loss_fn(batch_predictions, batch_targets)

            # ④ 计算梯度。
            batch_loss.backward()

            # ⑤ 根据梯度更新参数。
            optimizer.step()

            update_count += 1

        # 整轮训练完成后，计算验证集 MSE。
        validation_loss = evaluate(model, validation_loader, loss_fn)

        # ====================================================
        # 8. 验证结果更好时，保存当前最佳模型
        # ====================================================
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1

            # evaluate() 已经把模型切换到了评估模式。
            # 保存当前最佳参数对应的预测快照。
            with torch.no_grad():
                best_predictions = (model(check_inputs).detach().clone())

            # 立即写入文件，保留这一轮的参数。
            # 下次出现更好的模型时覆盖同一个文件。
            torch.save(model.state_dict(), model_path)

        # 每轮都验证，但只按日志间隔计算并打印训练集 MSE。
        if (epoch + 1) % config["log_interval"] == 0:
            train_loss = evaluate(
                model, train_eval_loader, loss_fn
            )

            print(
                f"epoch {epoch + 1:3d} | "
                f"累计更新 {update_count:4d} 次 | "
                f"训练 MSE = {train_loss:.6e} | "
                f"验证 MSE = {validation_loss:.6e}"
            )

    # ========================================================
    # 9. 训练结束，检查更新次数
    # ========================================================

    print(f"\n累计参数更新次数：{update_count}")
    print(f"最佳轮次：{best_epoch}")
    print(f"最佳验证 MSE：{best_validation_loss:.6e}")

    # ========================================================
    # 10. 读回配置和索引，检查文件内容是否一致
    # ========================================================
    loaded_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    loaded_indices = torch.load(
        split_path,
        map_location="cpu",
        weights_only=True,
    )

    assert loaded_config == config
    assert loaded_indices == split_indices

    print("通过：配置和划分索引保存、读取一致")

    # ========================================================
    # 11. 创建新模型，加载最佳参数
    # ========================================================

    # model 此时保留的是最后一轮参数。
    # loaded_model 将装入文件中保存的最佳轮次参数。
    loaded_model = nn.Linear(2, 1)

    saved_state = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    loaded_model.load_state_dict(saved_state)
    loaded_model.eval()

    # ========================================================
    # 12. 检查最佳参数保存前后的预测是否一致
    # ========================================================
    with torch.no_grad():
        loaded_predictions = loaded_model(check_inputs)

    # 必须至少成功记录过一次最佳模型。
    assert best_predictions is not None

    # 使用容差比较浮点数。
    assert torch.allclose(
        best_predictions,
        loaded_predictions,
        rtol=1e-5,
        atol=1e-8,
    )

    print("\n最佳模型保存前的预测：")
    print(best_predictions)
    print("加载后的预测：")
    print(loaded_predictions)
    print("通过：最佳模型保存前后的预测一致")

    # ========================================================
    # 13. 用加载后的最佳模型完成最终评估
    # ========================================================

    # 验证集用于确认加载后的表现。
    loaded_validation_loss = evaluate(
        loaded_model, validation_loader, loss_fn
    )

    # 测试集到这里才使用，不参与参数更新和最佳轮次选择。
    test_loss = evaluate(
        loaded_model, test_loader, loss_fn
    )

    print(f"\n加载后的验证 MSE：{loaded_validation_loss:.6e}")
    print(f"最终测试 MSE：{test_loss:.6e}")
    print("加载后的权重：", loaded_model.weight.detach())
    print("加载后的偏置：", loaded_model.bias.detach())

    print(f"\n配置保存位置：{config_path}")
    print(f"划分保存位置：{split_path}")
    print(f"参数保存位置：{model_path}")


# 直接运行这个文件时，执行 main()。
if __name__ == "__main__":
    main()