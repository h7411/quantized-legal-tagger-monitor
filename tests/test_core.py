"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Prepare a subset of the LEDGAR dataset for legal clause tagging.
def test_milestone_1():
    from app.core import (
        milestone_1,
        clean_clause,
        format_classification_prompt,
        LegalClauseTokenizer,
        LEDGAR_CATEGORIES,
    )
    res = milestone_1()
    assert res["status"] == "completed"
    assert res["train_count"] >= 16
    assert res["test_count"] >= 8
    assert res["num_categories"] == len(LEDGAR_CATEGORIES)
    assert res["vocab_size"] > 100
    assert "Clause:" in res["sample_prompt"]
    assert "Tag:" in res["sample_prompt"]

    # Test clean_clause
    dirty = "  This   Contract\r\nshall be  governed by   Delaware.  "
    assert clean_clause(dirty) == "This Contract shall be governed by Delaware."

    # Test format_classification_prompt
    p_with_tag = format_classification_prompt("Sample clause", "Termination")
    assert p_with_tag == "Clause: Sample clause\nTag: Termination"
    p_without_tag = format_classification_prompt("Sample clause")
    assert p_without_tag == "Clause: Sample clause\nTag:"

    # Test Tokenizer
    tok = res["tokenizer"]
    enc = tok.encode("Clause: Test\nTag: Governing Law")
    assert len(enc) > 0
    dec = tok.decode(enc)
    assert "Clause" in dec
    assert "Governing Law" in dec


# Milestone 2: Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.
def test_milestone_2():
    from app.core import (
        milestone_2,
        LegalGPT2Model,
        CausalSelfAttention,
        GPT2Block,
        predict_clause_tag,
    )
    import torch

    res = milestone_2(epochs=3)
    assert res["status"] == "completed"
    assert isinstance(res["model"], LegalGPT2Model)
    assert res["num_parameters"] > 10000
    assert len(res["loss_history"]) == 3
    # Check that training loss decreased
    assert res["final_loss"] <= res["initial_loss"] + 0.1
    assert res["predicted_tag"] == "Governing Law"

    # Test CausalSelfAttention unit behavior
    attn = CausalSelfAttention(d_model=32, n_head=2, max_seq_len=16)
    x = torch.randn(2, 6, 32)
    out_attn = attn(x)
    assert out_attn.shape == x.shape
    # Check causal mask property (upper triangular is 0)
    assert attn.mask[0, 0, 0, 1] == 0
    assert attn.mask[0, 0, 1, 0] == 1

    # Test GPT2Block
    block = GPT2Block(d_model=32, n_head=2)
    out_block = block(x)
    assert out_block.shape == x.shape


# Milestone 3: Evaluate the tagging accuracy of the fine-tuned model.
def test_milestone_3():
    from app.core import (
        milestone_3,
        compute_classification_metrics,
        LEDGAR_CATEGORIES,
    )

    res = milestone_3()
    assert res["status"] == "completed"
    assert res["total_evaluated"] >= 8
    assert res["accuracy"] >= 0.75
    assert res["precision"] >= 0.75
    assert res["recall"] >= 0.75
    assert res["f1_score"] >= 0.75
    assert len(res["per_class_metrics"]) == len(LEDGAR_CATEGORIES)
    assert len(res["sample_predictions"]) == res["total_evaluated"]

    # Test unit metrics function
    y_true = ["Governing Law", "Termination", "Termination"]
    y_pred = ["Governing Law", "Termination", "Governing Law"]
    m = compute_classification_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 2 / 3) < 1e-3
    assert m["total"] == 3
    assert m["correct"] == 2


# Milestone 4: Apply dynamic and static post-training quantization to the trained model.
def test_milestone_4():
    from app.core import (
        milestone_4,
        apply_dynamic_quantization,
        apply_static_quantization,
        StaticQuantizedLinear,
    )
    import torch
    import torch.nn as nn

    res = milestone_4()
    assert res["status"] == "completed"
    assert res["dynamic_quantized"] is True
    assert res["static_quantized"] is True
    assert res["dynamic_model"] is not None
    assert res["static_model"] is not None
    assert res["dynamic_pred"] == "Governing Law"
    assert res["static_pred"] == "Governing Law"

    # Test StaticQuantizedLinear module unit
    lin = nn.Linear(16, 8)
    sq = StaticQuantizedLinear(lin)
    assert sq.weight_int8.dtype == torch.int8
    x = torch.randn(2, 16)
    out = sq(x)
    assert out.shape == (2, 8)


# Milestone 5: Compare the original and quantized model sizes and measure simulated inference times.
def test_milestone_5():
    from app.core import (
        milestone_5,
        calculate_model_memory_footprint,
        simulate_inference_speed,
        LegalClauseTokenizer,
        LegalGPT2Model,
    )

    res = milestone_5()
    assert res["status"] == "completed"
    assert "benchmarks" in res
    bench = res["benchmarks"]
    assert "original" in bench
    assert "dynamic" in bench
    assert "static" in bench

    # Check size reduction
    assert bench["dynamic"]["size_bytes"] < bench["original"]["size_bytes"]
    assert bench["static"]["size_bytes"] < bench["original"]["size_bytes"]
    assert res["dynamic_compression_ratio"] > 1.2
    assert res["static_compression_ratio"] > 1.2

    # Test unit size calculation
    tok = LegalClauseTokenizer()
    m = LegalGPT2Model(vocab_size=tok.vocab_size)
    fp = calculate_model_memory_footprint(m)
    assert fp["size_bytes"] > 0
    assert fp["param_count"] > 10000

    # Test unit speed simulation
    speed = simulate_inference_speed(m, tok, ["This Agreement shall be governed by Delaware law."], num_runs=2, warmup=1)
    assert speed["mean_latency_ms"] > 0
    assert speed["throughput_clauses_per_sec"] > 0


# Milestone 6: Present a dashboard-like output of model efficiency metrics.
def test_milestone_6():
    from app.core import (
        milestone_6,
        format_efficiency_dashboard,
    )

    res = milestone_6()
    assert res["status"] == "completed"
    assert "dashboard" in res
    assert "QUANTIZED LEGAL TAGGER MONITOR" in res["dashboard"]
    assert "Original (FP32)" in res["dashboard"]
    assert "Dynamic INT8" in res["dashboard"]
    assert "Static INT8" in res["dashboard"]
    assert "RECOMMENDATION" in res["dashboard"]
    assert len(res["summary_table"]) == 3
    assert "recommendation" in res

    # Test format_efficiency_dashboard unit rendering
    mock_bench = {
        "original": {"size_kb": 700.0, "mean_latency_ms": 2.5, "throughput_clauses_per_sec": 400.0},
        "dynamic": {"size_kb": 350.0, "mean_latency_ms": 3.0, "compression_ratio": 2.0, "size_reduction_pct": 50.0, "throughput_clauses_per_sec": 330.0},
        "static": {"size_kb": 340.0, "mean_latency_ms": 2.8, "compression_ratio": 2.05, "size_reduction_pct": 51.4, "throughput_clauses_per_sec": 350.0},
    }
    mock_eval = {"accuracy": 0.95, "f1_score": 0.95}
    dash = format_efficiency_dashboard(mock_bench, mock_eval)
    assert "Original (FP32)" in dash
    assert "700.0" in dash
    assert "95.0%" in dash

