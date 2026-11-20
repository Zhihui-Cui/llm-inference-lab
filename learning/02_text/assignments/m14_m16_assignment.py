"""M14–M16 手写作业。每次只改一个 TODO，在命令行传入题号单独自测。"""

import argparse
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignments.m11_m13_assignment import TinyDecoder, make_optimizer, train_steps
from char_vocab import char_to_id
from m03_text.m04_sequence_windows import make_loader, train_samples
from m14_validation.m14_04_one_train_step import score_window
from m16_reload.m16_04_pack_checkpoint import model_config


def mean_token_loss(model, loader):
    """第 1 题。给同一个模型评分，返回所有目标位置的平均损失（Python 数字）。

    loader 每次给一批窗口，最后一批可能更小。窗口等长、无填充，至少有一批。
    score_window 已经计算了本批的平均损失。每个目标字在总平均里占相同分量。
    """
    total_loss = 0.0
    total_targets = 0
    for input_ids, target_ids in loader:
        batch_mean = score_window(model, input_ids, target_ids)
        target_count = target_ids.numel()
        # TODO 1：累计本批贡献的损失和题数，循环结束后返回平均值。
        raise NotImplementedError("请补全 mean_token_loss")


def train_one_epoch(model, optimizer, train_batches):
    """第 2 题。切到训练模式，按顺序走完所有训练批次，每批更新一次。

    可以复用 M13 的 train_steps。使用传入的模型和优化器。
    """
    raise NotImplementedError("请补全 train_one_epoch")


def greedy_ids(model, input_ids, max_new_tokens):
    """第 3 题。从单行 CPU long 张量 [1, 前缀长度] 开始，返回含前缀的完整编号张量。

    在推理模式下逐字取最高分，每次把完整的新前缀送入模型。
    总长度至多为位置表行数。前缀为空、超长或新增数为负时抛出 ValueError。
    """
    raise NotImplementedError("请补全 greedy_ids")


def restore_checkpoint(path):
    """第 4 题。在 CPU 读文件，返回已装入权重、切到 eval 的 TinyDecoder 和文件中的词表。

    本课使用固定的 TinyDecoder。用 model_config 核对配置，配置不一致时抛出 ValueError。
    """
    raise NotImplementedError("请补全 restore_checkpoint")


# 以下是分题自测。作答区在上面。
def check_mean():
    model = TinyDecoder()
    # 从训练长文本取前三个窗口。第一批有六道题，第二批有三道。
    samples = train_samples[:3]
    small_batches = make_loader(samples, batch_size=2, shuffle=False)
    all_inputs, all_targets = next(iter(make_loader(samples, batch_size=3, shuffle=False)))
    reference = score_window(model, all_inputs, all_targets)
    original = [p.detach().clone() for p in model.parameters()]
    result = mean_token_loss(model, small_batches)
    print("分两批评九道题", result, "一次评九道题", reference)
    assert math.isclose(result, reference, rel_tol=1e-6), "请核对每批的题数和最后的除数"
    assert all(torch.equal(old, p) for old, p in zip(original, model.parameters())), "评分改变了参数"


def check_epoch():
    train_batches = make_loader(train_samples[:3], batch_size=2, shuffle=False)
    model = TinyDecoder()
    model.eval()
    optimizer = make_optimizer(model)
    original = [p.detach().clone() for p in model.parameters()]
    steps = []
    hook = optimizer.register_step_post_hook(lambda *args: steps.append(model.training))
    try:
        train_one_epoch(model, optimizer, train_batches)
    finally:
        hook.remove()
    assert steps == [True, True], "三个窗口分成两批，各应在训练模式更新一次"
    assert any(not torch.equal(old, p) for old, p in zip(original, model.parameters())), "参数没有变化"


def check_greedy():
    model = TinyDecoder()
    prefix = torch.tensor([[char_to_id["我"], char_to_id["爱"]]])
    calls = []

    def record_forward(module, args, scores):
        assert not torch.is_grad_enabled() and not module.training, "生成时应使用 eval 和无梯度计算"
        calls.append((args[0].clone(), scores[0, -1].argmax().item()))

    hook = model.register_forward_hook(record_forward)
    try:
        result = greedy_ids(model, prefix.clone(), 3)
        expected = prefix[0].tolist()
        assert len(calls) == 3, "每生成一个编号就要做一次前向"
        for ids, chosen in calls:
            assert ids.tolist() == [expected], "请把刚选的编号接到前缀后再前向"
            expected.append(chosen)
        assert result.tolist() == [expected], "返回值要包含前缀和三个新增编号"
        capacity = model.position_embedding.num_embeddings
        almost_full = torch.zeros((1, capacity - 1), dtype=torch.long)
        calls.clear()
        assert greedy_ids(model, almost_full, 3).shape == (1, capacity)
        assert len(calls) == 1, "只剩一个空位，应只调用一次模型"
        calls.clear()
        assert torch.equal(greedy_ids(model, prefix, 0), prefix)
        for ids, limit in [(prefix[:, :0], 1), (torch.zeros((1, capacity + 1), dtype=torch.long), 1), (prefix, -1)]:
            try:
                greedy_ids(model, ids, limit)
            except ValueError:
                continue
            raise AssertionError("空前缀、超长前缀或负的新增数应报告 ValueError")
        assert not calls, "这些输入无需调用模型"
    finally:
        hook.remove()


def check_restore():
    reference = TinyDecoder().eval()
    # 改变两字的编号，检查是否真的使用了文件中的词表。
    vocab = dict(char_to_id)
    vocab["学"], vocab["助"] = vocab["助"], vocab["学"]
    bundle = dict(model_state_dict=reference.state_dict(), model_config=model_config(reference), char_to_id=vocab)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "test_checkpoint.pth"
        torch.save(bundle, path)
        loaded, loaded_vocab = restore_checkpoint(path)
        assert not loaded.training and loaded_vocab == vocab, "请核对 eval 和词表来源"
        for name, tensor in reference.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], tensor), f"{name} 与保存值不同"
        ids = torch.tensor([[vocab["我"], vocab["爱"]]])
        with torch.no_grad():
            torch.testing.assert_close(loaded(ids), reference(ids))
        bundle["model_config"]["num_heads"] += 1
        torch.save(bundle, path)
        try:
            restore_checkpoint(path)
        except ValueError:
            return
        raise AssertionError("请核对 checkpoint 配置")


if __name__ == "__main__":
    checks = {1: check_mean, 2: check_epoch, 3: check_greedy, 4: check_restore}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", type=int, nargs="?", choices=checks)
    question = parser.parse_args().question
    for number in ([question] if question else checks):
        torch.manual_seed(42)
        try:
            checks[number]()
        except NotImplementedError as error:
            sys.exit(f"第 {number} 题待完成。{error}")
        print(f"第 {number} 题通过")
