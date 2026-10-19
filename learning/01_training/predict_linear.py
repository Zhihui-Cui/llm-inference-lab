import torch
from pathlib import Path
from torch import nn

# 找到当前脚本同一目录下的参数文件。
model_path = Path(__file__).with_name("linear_regression_nn.pth")

load_model = nn.Linear(2, 1)

saved_state = torch.load(
    model_path,
    weights_only=True,
    map_location="cpu"
)

load_model.load_state_dict(saved_state)

load_model.eval()

inputs = torch.tensor([
    [ 3.0, -1.0],
    [ 0.0,  0.0],
    [-1.0,  2.0],
])

with torch.no_grad():
    predictions = load_model(inputs)

print(predictions)