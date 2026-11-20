"""用纯 Python 计算均方误差，并手动核对正常与异常案例。"""


def mse(y_true: list[float], y_pred: list[float]) -> float:
    """返回两个等长、非空数字列表的均方误差。

    长度不一致或列表为空时，抛出 ValueError。
    """
    if len(y_true) != len(y_pred):
        raise ValueError("真实值与预测值的长度必须一致")

    sample_count = len(y_true)
    if sample_count == 0:
        raise ValueError("输入列表不能为空")

    total_squared_error = 0.0
    for index in range(sample_count):
        total_squared_error += (y_true[index] - y_pred[index]) ** 2

    return total_squared_error / sample_count


if __name__ == "__main__":
    print("完全相等：", mse([1, 2, 3], [1, 2, 3]))  # 预期：0.0
    print("存在误差：", mse([1, 2, 3], [2, 2, 5]))  # 预期：5 / 3
    print("负数样本：", mse([-2], [1]))  # 预期：9.0

    # 分别捕获预期异常，让两个案例都能执行。
    for label, y_true, y_pred in [
        ("长度不一致", [1, 2], [1]),
        ("空列表", [], []),
    ]:
        try:
            result = mse(y_true, y_pred)
        except ValueError as error:
            print(f"{label}：捕获到预期的 ValueError（{error}）")
        else:
            raise AssertionError(
                f"{label}：预期抛出 ValueError，实际返回 {result}"
            )
