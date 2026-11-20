# Quantized Legal Tagger Monitor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/6ar4rnvk6j/badge.svg)](https://ask.thili.ai/b/6ar4rnvk6j)

This project develops a **Quantized Legal Tagger Monitor** for legal aid workers. It evaluates the size, memory footprint, and inference speed of a fine-tuned GPT-2 legal clause tagger, comparing original FP32 against dynamic and static post-training INT8 quantization to ensure rapid, private document processing on resource-constrained local laptops.

## What you are building

**Goal.** You will build a monitoring tool for legal aid workers to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems. This involves fine-tuning a GPT-2 model for legal clause classification and then applying quantization to measure its impact on model size and simulated performance.

**Done when.** The project is complete when you have a fine-tuned GPT-2 model capable of tagging legal clauses, and you have applied both dynamic and static post-training quantization to it. You must then present a dashboard-like output comparing the original and quantized models' sizes, simulated inference times, and tagging accuracy.

## Milestones Completed

- [x] **1. Prepare a subset of the LEDGAR dataset for legal clause tagging.**  
      Loaded, cleaned, and preprocessed representative contract provisions from the LEDGAR dataset across 8 legal categories (`Governing Law`, `Indemnification`, `Termination`, `Confidentiality`, `Amendments`, `Notices`, `Severability`, `Warranties`). Formatted clauses into classification-as-generation prompt format (`Clause: <text>\nTag: <label>`) and implemented the `LegalClauseTokenizer`.
- [x] **2. Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.**  
      Built a causal GPT-2 language model (`LegalGPT2Model`) with causal multi-head self-attention (`CausalSelfAttention`), pre-layer norm transformer blocks (`GPT2Block`), and token/positional embeddings. Fine-tuned on the LEDGAR subset using cross-entropy loss focused on the generated tag tokens.
- [x] **3. Evaluate the tagging accuracy of the fine-tuned model.**  
      Calculated standard evaluation metrics across test contract clauses: Accuracy, Macro/Weighted Precision, Macro/Weighted Recall, and F1-Score, along with per-category breakdowns.
- [x] **4. Apply dynamic and static post-training quantization to the trained model.**  
      Applied Dynamic Quantization (`apply_dynamic_quantization`) to quantize linear layers to INT8 on the fly. Built calibrated Static Post-Training Quantization (`apply_static_quantization`, `StaticQuantizedLinear`) to quantize both weights and activations to INT8 using calibration statistics.
- [x] **5. Compare the original and quantized model sizes and measure simulated inference times.**  
      Calculated model memory footprints in KB/MB and simulated inference speeds across original (FP32), dynamic (INT8), and static (INT8) variants, measuring mean latency, median latency, P95 latency, compression ratios, and throughput.
- [x] **6. Present a dashboard-like output of model efficiency metrics.**  
      Created an interactive ASCII comparative efficiency dashboard (`format_efficiency_dashboard`) and CLI output presenting size savings, latency comparisons, accuracy retention, and deployment recommendations for legal aid workers.

## Model Efficiency Dashboard

```
==============================================================================
          QUANTIZED LEGAL TAGGER MONITOR - MODEL EFFICIENCY DASHBOARD         
     Target Audience: Legal Aid Workers (Local Laptop & Resource-Constrained)  
==============================================================================
Metric                         | Original (FP32) | Dynamic INT8  | Static INT8  
------------------------------------------------------------------------------
Model Size (KB)                | 711.4         | 373.4         | 367.0        
Compression Ratio              | 1.00x         | 1.90x         | 1.94x        
Memory Reduction (%)           | 0.0%          | 47.5%         | 48.4%        
------------------------------------------------------------------------------
Latency per Clause (ms)        | 1.93          | 5.41          | 2.92         
Throughput (clauses/sec)       | 517.4         | 184.8         | 342.0        
------------------------------------------------------------------------------
Tagging Accuracy               | 100.0%        | 100.0%        | 100.0%       
Macro F1-Score                 | 100.0%        | 100.0%        | 100.0%       
==============================================================================
DEPLOYMENT RECOMMENDATION FOR LEGAL AID WORKERS:
  -> Deploy Static Quantized INT8 model on legal aid field laptops.
  -> Benefit: Saves ~48.4% memory while preserving 100% tagging accuracy.
  -> Enables fast, private, offline contract clause analysis on local devices.
==============================================================================
```
astrothunder
## Quickstart & CLI Usage

### 1. Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run CLI Commands

```bash
# Verify plumbing
python -m app.cli hello

# Milestone 1: Prepare LEDGAR dataset subset
python -m app.cli prepare-data

# Milestone 2: Fine-tune GPT-2 model
python -m app.cli train

# Milestone 3: Evaluate accuracy, precision, recall, and F1
python -m app.cli evaluate

# Milestone 4: Apply dynamic and static INT8 post-training quantization
python -m app.cli quantize

# Milestone 5: Benchmark model size and simulated inference latency
python -m app.cli benchmark

# Milestone 6: Display the comparative efficiency dashboard
python -m app.cli dashboard

# Classify a legal contract clause using the quantized model
python -m app.cli tag --clause "Each party shall hold in strict confidence all Proprietary Information." --quantized
```

### 3. Run the Test Suite

```bash
pytest -v
```

All 9 unit and smoke tests pass cleanly.

## CI/CD Pipeline

Both GitHub Actions (`.github/workflows/ci.yml`) and GitLab CI (`.gitlab-ci.yml`) pipelines are configured to run automated test workflows on Python 3.12.
