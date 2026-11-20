"""原 M11–M12 运行入口，供阶段记录和复核脚本重放。
分课练习从 m11_decoder/m11_decoder_block.py 开始。M13 当前练习见本目录 README.md。
"""

from char_vocab import char_to_id
from m11_decoder.m11_decoder_block import DecoderBlock


if __name__ == "__main__":
    from m11_decoder.m11_decoder_block import check_prefix
    from m11_decoder.m12_01_single_loss import inspect_single_loss
    from m11_decoder.m12_02_fixed_batch import TinyDecoder, fixed_examples, inspect_batch, mean_token_loss

    model = TinyDecoder()
    block, lm_head = model.block, model.lm_head
    inspect_single_loss(check_prefix(block), lm_head)

    inputs_batch, targets_batch = fixed_examples()
    x_batch = model.input_vectors(inputs_batch)
    logits_batch = lm_head(block(x_batch))
    logits_flat, targets_flat = inspect_batch(logits_batch, targets_batch)
    loss_batch = mean_token_loss(logits_batch, targets_batch)
    print("六个位置的平均损失：", loss_batch.item())
