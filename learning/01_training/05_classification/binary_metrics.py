import torch


def classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    """统计二分类结果：1 为有故障，0 为正常。"""
    if predictions.shape != targets.shape:
        raise ValueError("预测类别与真实标签的形状必须一致")

    if targets.numel() == 0:
        raise ValueError("输入不能为空")

    # 本轮先约定输入全部是 0、1 类别。
    true_positive = (
        (predictions == 1) & (targets == 1)
    ).sum().item()

    false_positive = (
        (predictions == 1) & (targets == 0)
    ).sum().item()

    false_negative = (
        (predictions == 0) & (targets == 1)
    ).sum().item()

    true_negative = (
        (predictions == 0) & (targets == 0)
    ).sum().item()

    sample_count = targets.numel()

    accuracy = (true_positive + true_negative) / sample_count

    # 模型一共报了多少台故障。
    predicted_positive_count = true_positive + false_positive

    # 实际一共有多少台故障。
    actual_positive_count = true_positive + false_negative

    # 本练习约定：分母为 0 时，指标返回 0.0。
    precision = (
        true_positive / predicted_positive_count
        if predicted_positive_count > 0
        else 0.0
    )

    recall = (
        true_positive / actual_positive_count
        if actual_positive_count > 0
        else 0.0
    )

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    # 前 10 台有故障，后 90 台正常。
    targets = torch.tensor([1] * 10 + [0] * 90)

    # 前 10 台：发现 8 台，漏掉 2 台。
    # 后 90 台：误报 12 台，正确排除 78 台。
    predictions = torch.tensor(
        [1] * 8 + [0] * 2 + [1] * 12 + [0] * 78
    )

    result = classification_metrics(predictions, targets)
    print(result)

    # 模型把所有设备都判为正常：没有任何报警。
    all_normal_predictions = torch.zeros_like(targets)

    print(classification_metrics(all_normal_predictions, targets))



if __name__ == "__main__":
    main()
