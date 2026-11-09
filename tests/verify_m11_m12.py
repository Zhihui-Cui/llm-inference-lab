"""Reproduce the M11–M12 forward checks on the published learning script."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import runpy
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.repo.resolve() / "learning/02_text/decoder_block.py"
    sys.path.insert(0, str(source.parent))
    sys.dont_write_bytecode = True
    torch.manual_seed(42)
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        demo = runpy.run_path(str(source), run_name="__main__")

    block = demo["block"].eval()
    x, logits, targets = (demo[name] for name in ("x_batch", "logits_batch", "targets_batch"))
    def error(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return (actual - expected).abs().max().item()

    with torch.no_grad():
        output = block(x)
        assert output.shape == x.shape == (2, 3, 2)
        batch_error = error(output, torch.stack([block(row) for row in x]))
        prefix_error = error(output[:, :2], block(x[:, :2]))
        # With both branches returning zero, the two residual paths pass x through.
        def zero_attention(module, inputs, result):
            assert result[0].shape == x.shape
            return torch.zeros_like(result[0]), result[1]
        def zero_ffn(module, inputs, result):
            assert result.shape == x.shape
            return torch.zeros_like(result)
        with block.attention.register_forward_hook(zero_attention), block.ffn.register_forward_hook(zero_ffn):
            residual_error = error(block(x), x)

        ids = demo["char_to_id"]
        expected = torch.tensor([[ids[ch] for ch in text] for text in ("爱学习", "码帮助")])
        assert torch.equal(targets, expected)
        assert logits.shape == (2, 3, len(ids)) and demo["logits_flat"].shape == (6, len(ids))
        assert torch.equal(demo["targets_flat"], expected.reshape(-1))
        # Independently select each correct target from its own [sentence, position] row.
        log_probs = logits.log_softmax(dim=-1)
        token_losses = -log_probs.gather(-1, expected.unsqueeze(-1)).squeeze(-1)
        ce_error = error(demo["loss_batch"], token_losses.mean())
        assert torch.isfinite(token_losses).all()

    evidence = {
        "executor": "assistant", "status": "passed", "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__, "device": str(x.device),
        "dtype": str(x.dtype), "seed_for_this_verification": 42, "rtol": 1e-5, "atol": 1e-6,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (source, source.with_name("char_vocab.py"))},
        "shapes": {"block_input": list(x.shape), "block_output": list(output.shape),
                   "logits": list(logits.shape), "flat_scores": list(demo["logits_flat"].shape),
                   "flat_targets": list(demo["targets_flat"].shape)},
        "max_abs_errors": {"batch_vs_separate": batch_error, "prefix": prefix_error,
                           "zero_branches_vs_input": residual_error, "ce_vs_selected_log_probs": ce_error},
        "target_characters": ["爱学习", "码帮助"], "targets_flat": expected.reshape(-1).tolist(),
        "batch_loss": demo["loss_batch"].item(), "training_updates": 0, "entrypoint_stdout": stdout.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: evidence[k] for k in ("status", "shapes", "max_abs_errors", "batch_loss")}))


if __name__ == "__main__":
    main()
