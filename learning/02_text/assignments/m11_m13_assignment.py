"""M11–M13 综合手写作业｜学完固定 batch 短训练以后打开。

目标：从“我爱学习”和“代码帮助”出发，自己接通 decoder 的训练链路。
只在“手写区”作答。可使用 PyTorch 的标准层、损失和优化器，以及以前的字符词表。
第一次尝试时合上前面的练习。遇到卡点再查阅，并记下你查过哪一步。

在这个文件所在目录运行：python m11_m13_assignment.py 1
通过后把 1 改为 2、3、4，分别检查各题。省略数字会从头检查，停在第一道未完成的题。
完成后请用自己的数值解释：哪几行位置有梯度？更新是怎样发生的？
这个 batch 的 loss 下降能说明什么？想检验新句子上的效果，还需要什么数据？
"""

import argparse

import torch
from torch import nn

# 直接运行本课时，让 Python 也能找到 02_text 中前面学过的模块。
if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from char_vocab import char_to_id, encode


# ─── 手写区 ───

class DecoderBlock(nn.Module):
    """第 1 题。输入 [句子数, 位置数, 特征数]，输出形状相同。

    写一个 pre-norm block。使用因果自注意力、两次残差和两层逐位置 FFN。
    注意力 dropout 为 0。框架会传入特征数 2、FFN 宽度 4、注意力头数 2。
    """

    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        z = self.norm_attn(x)
        T = x.shape[-2]
        blocked = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        attn_out, _ = self.attention(z, z, z, attn_mask=blocked, need_weights=False)
        h = x + attn_out
        ffn_input = self.norm_ffn(h)
        ffn_out = self.ffn(ffn_input)
        return ffn_out + h


def make_batch(sentences):
    """第 2 题。输入若干等长句子，返回两个 CPU long 张量。

    每行来自同一句。第一个张量存输入字的 ID，第二个存每个位置的下一个字 ID。
    本作业每句至少两个字，无 padding。可以复用 encode(句子, char_to_id)。
    """
    text = torch.tensor(
        [encode(sentence, char_to_id) for sentence in sentences],
        dtype=torch.long,
    )

    inputs_batch = text[:, :-1]
    targets_batch = text[:, 1:]

    return inputs_batch, targets_batch


class TinyDecoder(nn.Module):
    """第 3 题。四个部件已给出，请把它们连接成一次前向。"""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(len(char_to_id), 2)
        self.position_embedding = nn.Embedding(8, 2)
        self.block = DecoderBlock(d_model=2, d_ff=4, num_heads=2)
        self.lm_head = nn.Linear(2, len(char_to_id))

    def forward(self, input_ids):
        """从 [句子数, 位置数] 的字符 ID 算出 [句子数, 位置数, 词表大小] 的原始分数。"""
        position_idx = torch.arange(input_ids.shape[1])
        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(position_idx)
        x = token_vectors + position_vectors
        hidden = self.block(x)
        scores = self.lm_head(hidden)
        return scores


def batch_loss(scores, target_ids):
    """第 3 题。scores 是原始分数，target_ids 是下一字编号，返回所有位置的平均 CE 标量。"""
    new_scores = scores.reshape(-1, scores.shape[-1])
    new_target_ids = target_ids.reshape(-1)
    loss = nn.functional.cross_entropy(new_scores, new_target_ids)
    return loss


def make_optimizer(model):
    """第 4 题。为传入的模型创建 SGD，学习率 0.1，动量和权重衰减均为 0。"""
    return torch.optim.SGD(model.parameters(), lr=0.1)


def train_steps(model, optimizer, input_ids, target_ids, steps):
    """第 4 题。使用传入的模型和优化器，对固定的 ID 做 steps 次更新。

    返回 Python 数字列表。第 0 项是开始时的平均 loss，第 k 项是第 k 次更新
    后从同一批 ID 重新前向测得的平均 loss。多次调用时接着已有权重训练。
    """
    scores = model(input_ids)
    loss = batch_loss(scores, target_ids)
    history = [loss.item()]
    for _ in range(steps):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scores = model(input_ids)
        loss = batch_loss(scores, target_ids)
        history.append(loss.item())
    return history


# ─── 自测区。下面的代码用于批改，可以按题号运行。 ───

SENTENCES = ("我爱学习", "代码帮助")


def check_block():
    block = DecoderBlock(2, 4, 2)
    x = torch.tensor([[[2., 4.], [14., 10.], [3., 6.]]])
    output = block(x)
    assert output.shape == x.shape, "block 要保留每个位置的两个特征"
    torch.testing.assert_close(output[:, :2], block(x[:, :2]))
    # 翻转首字的两个特征，检查后面的位置是否能够读取前文。
    changed = x.clone()
    changed[:, 0] = x[:, 0].flip(-1)
    assert not torch.allclose(block(changed)[:, 2], output[:, 2]), "检查最后一个位置能否读取首字"
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    torch.testing.assert_close(block(x), x, msg="两条分支都输出零时，残差应保留原输入")


def check_batch():
    for sentences in (SENTENCES, ("我爱", "学习")):
        input_ids, target_ids = make_batch(sentences)
        for row, sentence in enumerate(sentences):
            torch.testing.assert_close(input_ids[row], torch.tensor(encode(sentence[:-1], char_to_id)))
            torch.testing.assert_close(target_ids[row], torch.tensor(encode(sentence[1:], char_to_id)))
        assert input_ids.shape == target_ids.shape == (len(sentences), len(sentences[0]) - 1)


def check_loss():
    model = TinyDecoder()
    input_ids, target_ids = make_batch(SENTENCES)
    scores = model(input_ids)
    assert scores.shape == (2, 3, len(char_to_id))
    # 裁去未来位置，保留位置的预测应一致。两句话单独前向也应和批次内一致。
    torch.testing.assert_close(scores[:, :2], model(input_ids[:, :2]))
    torch.testing.assert_close(scores[1:2], model(input_ids[1:2]))
    loss = batch_loss(scores, target_ids)
    correct_scores = scores.log_softmax(-1).gather(-1, target_ids.unsqueeze(-1))
    torch.testing.assert_close(loss, -correct_scores.mean())
    loss.backward()
    gradient = model.position_embedding.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all(), "检查位置表的梯度"
    assert (gradient[:3].abs().sum(dim=1) > 0).all(), "当前 batch 用了位置编号 0、1、2"
    assert torch.count_nonzero(gradient[3:]) == 0, "检查未使用的位置"
    print("位置表的梯度", gradient.tolist())


def check_training():
    model = TinyDecoder()
    input_ids, target_ids = make_batch(SENTENCES)
    optimizer = make_optimizer(model)
    assert isinstance(optimizer, torch.optim.SGD), "本题用 SGD 核对一次实际更新"
    covered = [p for group in optimizer.param_groups for p in group["params"]]
    assert sorted(map(id, covered)) == sorted(map(id, model.parameters())), "检查优化器覆盖的参数"

    # 在每次真实的 step 之前读取梯度，便于定位断开的部件和非有限值。
    steps_seen = 0

    def check_gradients(opt, args, kwargs):
        nonlocal steps_seen
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} 没收到梯度"
            assert torch.isfinite(parameter.grad).all(), f"{name} 的梯度有非有限值"
        steps_seen += 1

    with torch.no_grad():
        initial_loss = batch_loss(model(input_ids), target_ids).item()
        initial_weights = {name: value.clone() for name, value in model.named_parameters()}
    probe = model.position_embedding.weight
    before = probe[0, 0].item()
    first_gradient = torch.autograd.grad(batch_loss(model(input_ids), target_ids), probe)[0][0, 0].item()
    with optimizer.register_step_pre_hook(check_gradients):
        first = train_steps(model, optimizer, input_ids, target_ids, 1)
        after = probe[0, 0].item()
        assert abs(after - (before - 0.1 * first_gradient)) < 1e-6, "检查位置表 [0, 0] 的一步更新"
        assert after != before and steps_seen == 1, "检查实际更新次数和参数变化"
        for group_name, group in model.named_children():
            assert any(
                not torch.equal(value, initial_weights[f"{group_name}.{name}"])
                for name, value in group.named_parameters()
            ), f"检查 {group_name} 的参数是否实际变化"
        rest = train_steps(model, optimizer, input_ids, target_ids, 79)
    assert steps_seen == 80, "请复用传入的优化器，按指定步数更新"
    assert len(first) == 2 and len(rest) == 80, "每段记录都需要包括更新前和最后一步的 loss"
    assert torch.isfinite(torch.tensor(first + rest)).all(), "loss 中出现了非有限值"
    assert abs(first[0] - initial_loss) < 1e-6 and abs(first[-1] - rest[0]) < 1e-6
    with torch.no_grad():
        final_loss = batch_loss(model(input_ids), target_ids).item()
    assert abs(rest[-1] - final_loss) < 1e-6, "最后一项应使用更新后的参数重新评分"
    assert final_loss < 0.9 * initial_loss, "本作业固定种子的 80 步检查要求平均 loss 下降超过 10%"
    print("位置表 [0, 0] 更新前、梯度、更新后", before, first_gradient, after)
    print("第 0、1、80 步的 loss", first[0], first[-1], final_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按题号运行 M11–M13 手写作业自测")
    parser.add_argument("exercise", type=int, nargs="?", choices=range(1, 5), help="题号 1 至 4")
    exercise = parser.parse_args().exercise
    torch.set_num_threads(1)
    for number, check in enumerate((check_block, check_batch, check_loss, check_training), start=1):
        if exercise is None or exercise == number:
            torch.manual_seed(42)
            try:
                check()
            except NotImplementedError as pending:
                raise SystemExit(f"待手写 · {pending}")
            print(f"第 {number} 题通过")
