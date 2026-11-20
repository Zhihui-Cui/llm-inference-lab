import torch
from torch import nn

def evaluate(model, data_loader, loss_fn) -> float:
    model.eval()

    total_error_sum = 0.0
    total_sample = 0

    with torch.no_grad():

        for batch_inputs, batch_targets in data_loader:
            batch_prediction = model(batch_inputs)
            batch_loss = loss_fn(batch_prediction, batch_targets)

            total_error_sum += batch_loss.item() * batch_inputs.shape[0]
            total_sample += batch_inputs.shape[0]

    return total_error_sum / total_sample

def train_and_save_best(config, model, validation_loader, loss_fn, model_path):
    # 放在训练循环之前：
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(config["num_epochs"]):
        # 此处省略已有的训练代码。

        validation_loss = evaluate(model, validation_loader, loss_fn)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

def load_model(model_path, dataloader, loss_fn):
    # 1. 创建结构相同的新模型。
    loaded_model = nn.Linear(2, 1)

    # 2. 从文件读取参数字典。
    model_state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    # 3. 把参数装入新模型。
    loaded_model.load_state_dict(model_state_dict)

    # 4. evaluate 内部负责切换评估模式、关闭梯度记录。
    test_loss = evaluate(loaded_model, dataloader, loss_fn)

    # 5. 打印测试结果。
    print(f"加载模型后的测试 MSE：{test_loss:.6e}")


def main(config, model, train_loader, validation_loader, optimizer, loss_fn, model_path, dataloader):
    best_validation_loss = float("inf")
    best_epoch = 0

    for epoch in range(config["num_epochs"]):
        model.train()

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)

            loss.backward()
            optimizer.step()

        validation_loss = evaluate(model, validation_loader, loss_fn)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1

            torch.save(model.state_dict(), model_path)

    print(
        f"最佳轮次：{best_epoch}"
        f"最佳验证MSE：{best_validation_loss:.6e}"
    )

    loaded_model = nn.Linear(2, 1)

    loaded_state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    loaded_model.load_state_dict(loaded_state_dict)

    test_loss = evaluate(loaded_model, dataloader, loss_fn)

    print(f"加载模型后的测试 MSE：{test_loss:.6e}")
