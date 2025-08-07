import torch
import sentencepiece as spm
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from speculative_prefill.hf_speculative_prefill import HFFSpeculativePrefill
from unittest.mock import patch


def build_tiny_llama(save_dir):
    text_path = save_dir / "train.txt"
    text_path.write_text("a b c d e f g h i j")
    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(save_dir / "sp"),
        vocab_size=24,
        model_type="bpe",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
    )
    tokenizer = LlamaTokenizer(str(save_dir / "sp.model"))
    tokenizer.save_pretrained(save_dir)
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(save_dir)
    return save_dir


def test_speculative_prefill_matches_base(tmp_path):
    model_dir = build_tiny_llama(tmp_path)
    prefill = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=1,
        keep_percentage=1.0,
        tensor_parallel=False,
        device="cpu",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a b")
    logits_spec = prefill(prompt_path)
    logits_base = prefill.base_model(
        prefill.last_final_ids, position_ids=prefill.last_position_ids
    ).logits
    assert torch.allclose(logits_spec, logits_base)


def test_disable_speculation(tmp_path):
    model_dir = build_tiny_llama(tmp_path)
    prefill = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=1,
        keep_percentage=1.0,
        tensor_parallel=False,
        device="cpu",
        speculative=False,
    )
    assert prefill.spec_model is None
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a b")
    prompt_text = prompt_path.read_text()
    logits = prefill(prompt_path)
    base_ids = prefill.tokenizer(prompt_text, return_tensors="pt").input_ids
    pos_ids = torch.arange(0, base_ids.size(1)).unsqueeze(0)
    logits_base = prefill.base_model(base_ids, position_ids=pos_ids).logits
    assert torch.allclose(logits, logits_base)


def test_measure_ttft(tmp_path, capsys):
    model_dir = build_tiny_llama(tmp_path)
    prefill_spec = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=1,
        keep_percentage=1.0,
        tensor_parallel=False,
        device="cpu",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a b")
    spec_time = prefill_spec.measure_ttft(prompt_path)
    captured_spec = capsys.readouterr().out
    spec_lines = captured_spec.strip().splitlines()
    assert any(line.startswith("TTFT (spec):") for line in spec_lines)
    gen_spec = prefill_spec.base_model.generate(
        prefill_spec.last_final_ids, max_new_tokens=16, do_sample=False
    )[0]
    expected_spec = prefill_spec.tokenizer.decode(
        gen_spec[prefill_spec.last_final_ids.size(1) :],
        skip_special_tokens=True,
    )
    assert spec_lines[-1] == expected_spec
    assert isinstance(spec_time, float)

    prefill_base = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=1,
        keep_percentage=1.0,
        tensor_parallel=False,
        device="cpu",
        speculative=False,
    )
    base_time = prefill_base.measure_ttft(prompt_path)
    captured_base = capsys.readouterr().out
    base_lines = captured_base.strip().splitlines()
    assert any(line.startswith("TTFT (no spec):") for line in base_lines)
    gen_base = prefill_base.base_model.generate(
        prefill_base.last_final_ids, max_new_tokens=16, do_sample=False
    )[0]
    expected_base = prefill_base.tokenizer.decode(
        gen_base[prefill_base.last_final_ids.size(1) :],
        skip_special_tokens=True,
    )
    assert base_lines[-1] == expected_base
    assert isinstance(base_time, float)


def test_speculator_runs_once_when_zero_steps(tmp_path):
    model_dir = build_tiny_llama(tmp_path)
    prefill = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=0,
        keep_percentage=1.0,
        tensor_parallel=False,
        device="cpu",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a b")
    with patch.object(prefill.spec_model, "forward", wraps=prefill.spec_model.forward) as fwd:
        prefill(prompt_path)
        assert fwd.call_count == 1


def test_measure_ttft_zero_lookahead(tmp_path, capsys):
    model_dir = build_tiny_llama(tmp_path)
    prefill = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=0,
        keep_percentage=1.0,
        tensor_parallel=False,
        device="cpu",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a b")
    ttft = prefill.measure_ttft(prompt_path)
    output = capsys.readouterr().out
    lines = output.strip().splitlines()
    assert any(line.startswith("TTFT (spec):") for line in lines)
    gen = prefill.base_model.generate(
        prefill.last_final_ids, max_new_tokens=16, do_sample=False
    )[0]
    expected = prefill.tokenizer.decode(
        gen[prefill.last_final_ids.size(1) :], skip_special_tokens=True
    )
    assert lines[-1] == expected
    assert isinstance(ttft, float)


def test_prefill_prints_pruned_tokens(tmp_path, capsys):
    model_dir = build_tiny_llama(tmp_path)
    prefill = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=1,
        keep_percentage=0.5,
        tensor_parallel=False,
        device="cpu",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a b c d")
    prefill(prompt_path)
    out = capsys.readouterr().out
    assert "Speculative prefill output:" in out


def test_last_prompt_token_preserved(tmp_path):
    model_dir = build_tiny_llama(tmp_path)
    prefill = HFFSpeculativePrefill(
        str(model_dir),
        str(model_dir),
        look_ahead=1,
        keep_percentage=0.25,
        tensor_parallel=False,
        device="cpu",
    )
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("a b c d e")
    prefill(prompt_path)
    prompt_text = prompt_path.read_text()
    base_ids = prefill.tokenizer(prompt_text, return_tensors="pt").input_ids
    seq_len = base_ids.size(1)
    keep_mask = prefill.last_position_ids[0] < seq_len
    kept_positions = prefill.last_position_ids[0][keep_mask]
    assert (seq_len - 1) in kept_positions

