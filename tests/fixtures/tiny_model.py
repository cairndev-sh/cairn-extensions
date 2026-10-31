#!/usr/bin/env python3
"""tests/fixtures/tiny_model.py — build a throwaway Llama-style model and tokenizer on disk, offline.

    python tests/fixtures/tiny_model.py <output-dir> [--data <jsonl>]

~0.1M random-weight parameters and a 512-token BPE tokenizer trained on the demo's sample rows. It
writes sentences of nothing, which is the point: it lets `demos/laptop/run.sh --tiny` prove
the whole governed path (prepare, LoRA, held-out loss, gates, promote) in about a minute with no
download. The default run uses a real small model from the Hugging Face Hub instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "packs" / "cairn-pack-dataset-prep" / "data" / "sample-ifc.jsonl"
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def build(out: Path, data: Path, vocab_size: int = 512, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus = [f"{r.get('instruction', '')}\n{r.get('input', '')}\n{r.get('output', '')}" for r in rows]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=SPECIALS[0], pad_token=SPECIALS[0],
                                   bos_token=SPECIALS[0])
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out)

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                         bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
                         pad_token_id=fast.pad_token_id, tie_word_embeddings=True)
    LlamaForCausalLM(config).save_pretrained(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = p.parse_args()
    print(build(args.out, args.data))


if __name__ == "__main__":
    main()
