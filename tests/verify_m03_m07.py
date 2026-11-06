"""Assistant acceptance checks for the guided M03-M07 exercises.

Run from the archived repository: python tests/verify_m03_m07.py
Use --repo to check another checkout, and --output to save the JSON evidence.
This verifies behavior; it does not assess the learner's independent mastery.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import runpy
import subprocess
import sys

import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source_dir = repo / "learning" / "02_text"
    names = ["char_vocab.py", "sequence_windows.py", "attention_practice.py", "trainable_attention.py"]
    report = {
        "schema_version": 1,
        "executor": "assistant",
        "purpose": "Behavioral acceptance of guided M03-M07 exercises, not independent learner mastery",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "float32"},
        "seed": 42,
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "source_sha256": {},
        "demo_scripts": [],
        "checks": [],
        "scope_limits": [
            "Tiny authored text fixtures, not a sourced public corpus or held-out evaluation benchmark.",
            "Attention classification demo trains only its output layer on fixed features.",
            "M07 exercises numerical inputs and an MSE target, not end-to-end text language-model training.",
            "This assistant-run audit does not establish independent implementation or lasting conceptual mastery.",
        ],
    }
    sys.path.insert(0, str(source_dir))
    modules = {}
    for name in names:
        path = source_dir / name
        report["source_sha256"][f"learning/02_text/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        completed = subprocess.run([sys.executable, "-B", str(path)], cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        report["demo_scripts"].append({"path": f"learning/02_text/{name}", "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
        # Capture educational top-level prints while obtaining the actual functions.
        with contextlib.redirect_stdout(io.StringIO()):
            modules[name] = runpy.run_path(str(path))

    def check(module, name, action):
        try:
            details = action()
            report["checks"].append({"module": module, "name": name, "status": "pass", "details": details})
        except Exception as exc:
            report["checks"].append({"module": module, "name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"})

    def close(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    vocab = modules["char_vocab.py"]
    windows = modules["sequence_windows.py"]
    attention = modules["attention_practice.py"]
    trainable = modules["trainable_attention.py"]

    def vocab_check():
        mapping = vocab["char_to_id"]
        reverse = vocab["id_to_char"]
        train = vocab["train_text"]
        assert set(mapping) == set(train) | {"<unk>"}
        assert len(set(mapping.values())) == len(mapping)
        assert mapping["<unk>"] == 0
        assert vocab["decode"](vocab["encode"](train, mapping), reverse) == train
        unknowns = set(vocab["validation_text"] + vocab["test_text"]) - set(train)
        assert unknowns and all(char not in mapping for char in unknowns)
        assert all(vocab["encode"](char, mapping) == [0] for char in unknowns)
        decoded = {split: vocab["decode"](vocab["encode"](vocab[f"{split}_text"], mapping), reverse) for split in ("validation", "test")}
        assert decoded == {"validation": "我爱<unk>。", "test": "我爱<unk>。"}
        assert vocab["encode"]("", mapping) == [] and vocab["decode"]([], reverse) == ""
        return {"vocab_size_including_unk": len(mapping), "train_characters": len(train), "held_out_unknown_characters": sorted(unknowns), "held_out_decoded": decoded}

    check("M03", "training_only_vocabulary_roundtrip_and_unknowns", vocab_check)

    def window_check():
        result = {}
        for split in ("train", "validation", "test"):
            ids = windows[f"{split}_ids"]
            samples = windows[f"{split}_samples"]
            assert len(samples) == max(len(ids) - 3, 0)
            for start, (inputs, targets) in enumerate(samples):
                assert inputs == ids[start:start + 3]
                assert targets == ids[start + 1:start + 4]
                assert inputs[1:] == targets[:-1]
            stored_inputs, stored_targets = windows[f"{split}_loader"].dataset.tensors
            assert stored_inputs.dtype == stored_targets.dtype == torch.long
            assert stored_inputs.tolist() == [pair[0] for pair in samples]
            assert stored_targets.tolist() == [pair[1] for pair in samples]
            shapes = [list(inputs.shape) for inputs, targets in windows[f"{split}_loader"]]
            assert sum(shape[0] for shape in shapes) == len(samples)
            assert all(shape[1] == 3 for shape in shapes)
            result[split] = {"token_count": len(ids), "window_count": len(samples), "batch_shapes": shapes, "first_window": samples[0]}
        assert windows["make_windows"]([1, 2, 3], 3) == []
        assert windows["make_windows"]([1, 2, 3, 4], 3) == [([1, 2, 3], [2, 3, 4])]
        try:
            windows["make_loader"]([], 2, False)
        except ValueError:
            result["empty_loader_rejected"] = True
        else:
            raise AssertionError("Empty samples must be rejected")
        return result

    check("M04", "shifted_windows_stay_within_each_split_and_keep_last_batch", window_check)

    q, k, v = (attention[name] for name in ("queries", "keys", "values"))

    def hand_attention_check():
        # Independent scalar arithmetic for Q=[3,1], K rows [2,1],[1,2],[1,1].
        raw = [7.0, 5.0, 4.0]
        scaled = [score / math.sqrt(2.0) for score in raw]
        exponentials = [math.exp(score - max(scaled)) for score in scaled]
        weights = [value / math.fsum(exponentials) for value in exponentials]
        output = [math.fsum(weights[j] * float(v[j, feature]) for j in range(3)) for feature in range(2)]
        actual_outputs, actual_weights = attention["single_head_attention"](q, k, v)
        close(q[0] @ k.T, torch.tensor(raw))
        close(actual_weights[0], torch.tensor(weights))
        close(actual_outputs[0], torch.tensor(output))
        close(actual_weights.sum(-1), torch.ones(3))
        for i in range(3):
            one_output, one_weights = attention["attend_one_position"](q[i], k, v)
            close(actual_outputs[i], one_output)
            close(actual_weights[i], one_weights)
        return {"hand_scores_row_1": raw, "hand_weights_row_1": weights, "hand_output_row_1": output, "matrix_output_row_1": actual_outputs[0].tolist(), "row_weight_sums": actual_weights.sum(-1).tolist()}

    check("M05", "independent_scalar_reference_and_loop_matrix_equivalence", hand_attention_check)

    def value_change_check():
        before, weights = attention["single_head_attention"](q, k, v)
        changed_v = v.clone()
        changed_v[1, 1] += 10.0
        after, changed_weights = attention["single_head_attention"](q, k, changed_v)
        close(weights, changed_weights)
        expected_delta = weights[:, 1:2] * torch.tensor([[0.0, 10.0]])
        close(after - before, expected_delta)
        assert torch.equal(before[:, 0], after[:, 0])
        return {"perturbation": "V at source position 2, feature 2 += 10", "weight_change_max": float((changed_weights - weights).abs().max()), "output_delta": (after - before).tolist(), "predicted_delta": expected_delta.tolist()}

    check("M05", "changing_v_preserves_weights_and_changes_only_its_feature", value_change_check)

    def causal_check():
        changed_k, changed_v = k.clone(), v.clone()
        changed_k[2] += 10
        changed_v[2] += 10
        before, weights = attention["causal_attention"](q, k, v)
        after, _ = attention["causal_attention"](q, changed_k, changed_v)
        close(before[:2], after[:2])
        assert torch.equal(weights.triu(1), torch.zeros_like(weights))
        close(weights.sum(-1), torch.ones(3))
        plain_before, _ = attention["single_head_attention"](q, k, v)
        plain_after, _ = attention["single_head_attention"](q, changed_k, changed_v)
        assert not torch.allclose(plain_before[:2], plain_after[:2], rtol=1e-5, atol=1e-6)
        one_output, one_weight = attention["causal_attention"](q[:1], k[:1], v[:1])
        assert torch.isfinite(one_output).all() and torch.isfinite(one_weight).all()
        close(one_output, v[:1])
        close(one_weight, torch.ones(1, 1))
        return {"causal_prefix_delta_max": float((after[:2] - before[:2]).abs().max()), "unmasked_prefix_delta_max": float((plain_after[:2] - plain_before[:2]).abs().max()), "t1_weight": one_weight.tolist(), "t1_output": one_output.tolist()}

    check("M06", "causal_future_perturbation_unmasked_control_and_single_token", causal_check)

    def trainable_check():
        torch.manual_seed(42)
        layers = [nn.Linear(2, 2, bias=False) for _ in range(3)]
        single = torch.tensor([[2., 1.], [1., 2.], [1., 1.]])
        inputs = torch.stack([single, single + 1]).requires_grad_()

        def forward(x):
            return trainable["causal_attention"](*trainable["make_qkv"](x, *layers))

        outputs, weights = forward(inputs)
        assert list(outputs.shape) == [2, 3, 2] and list(weights.shape) == [2, 3, 3]
        assert torch.isfinite(outputs).all() and torch.isfinite(weights).all()
        close(weights.sum(-1), torch.ones(2, 3))
        for index in range(2):
            separate_output, separate_weights = forward(inputs[index])
            close(outputs[index], separate_output)
            close(weights[index], separate_weights)
        changed_inputs = inputs.detach().clone()
        changed_inputs[1] += 10
        changed_output, _ = forward(changed_inputs)
        close(changed_output[0], outputs[0])
        assert not torch.allclose(changed_output[1], outputs[1])
        input_grad = torch.autograd.grad(outputs[0].sum(), inputs, retain_graph=True)[0]
        assert torch.equal(input_grad[1], torch.zeros_like(input_grad[1]))
        assert float(input_grad[0].abs().sum()) > 0
        parameter_before = [layer.weight.detach().clone() for layer in layers]
        target = torch.tensor([[1., 0.], [0., 1.]])
        loss = nn.functional.mse_loss(outputs[:, -1], target)
        loss.backward()
        grad_norms = {}
        for name, layer, original in zip(("Q", "K", "V"), layers, parameter_before):
            assert torch.equal(original, layer.weight)
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
            grad_norms[name] = float(layer.weight.grad.norm())
            assert grad_norms[name] > 0
        optimizer = torch.optim.SGD([parameter for layer in layers for parameter in layer.parameters()], lr=0.03)
        optimizer.step()
        deltas = {name: float((layer.weight.detach() - original).abs().max()) for name, layer, original in zip(("Q", "K", "V"), layers, parameter_before)}
        assert all(delta > 0 for delta in deltas.values())
        with torch.no_grad():
            updated, _ = forward(inputs)
            updated_loss = nn.functional.mse_loss(updated[:, -1], target)
        assert torch.isfinite(updated_loss)
        assert updated_loss < loss
        return {"output_shape": list(outputs.shape), "weights_shape": list(weights.shape), "unmodified_sample_delta_max": float((changed_output[0] - outputs[0]).detach().abs().max()), "modified_sample_delta_max": float((changed_output[1] - outputs[1]).detach().abs().max()), "other_sample_input_gradient_max": float(input_grad[1].abs().max()), "projection_gradient_norms": grad_norms, "backward_alone_kept_parameters_unchanged": True, "optimizer_parameter_delta_max": deltas, "batch_fixture_loss_before": float(loss.detach()), "batch_fixture_loss_after": float(updated_loss), "original_demo_loss_before": float(trainable["loss"].detach()), "original_demo_loss_after": float(trainable["new_loss"].detach())}

    check("M07", "batch_isolation_shapes_finite_gradients_and_optimizer_boundary", trainable_check)
    report["status"] = "pass" if all(item["status"] == "pass" for item in report["checks"]) and all(item["exit_code"] == 0 for item in report["demo_scripts"]) else "fail"
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "checks": len(report["checks"]), "demo_scripts": len(report["demo_scripts"]), "output": str(args.output)}, ensure_ascii=False))
    else:
        print(serialized)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
