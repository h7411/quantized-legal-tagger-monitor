# Quantized Legal Tagger Monitor

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/6ar4rnvk6j/badge.svg)](https://ask.thili.ai/b/6ar4rnvk6j)

This project develops a 'Quantized Legal Tagger Monitor' for legal aid workers. It allows them to assess the efficiency of a legal clause tagger, specifically focusing on how quantization affects its size and inference speed. The goal is to ensure legal document processing remains efficient on local systems without sacrificing critical accuracy.

## What you are building

**Goal.** You will build a monitoring tool for legal aid workers to evaluate the size and inference speed of a quantized legal clause tagger, ensuring efficient document processing on local systems. This involves fine-tuning a GPT-2 model for legal clause classification and then applying quantization to measure its impact on model size and simulated performance.

**Done when.** The project is complete when you have a fine-tuned GPT-2 model capable of tagging legal clauses, and you have applied both dynamic and static post-training quantization to it. You must then present a dashboard-like output comparing the original and quantized models' sizes, simulated inference times, and tagging accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a subset of the LEDGAR dataset for legal clause tagging.**  
      Hint: Focus on data loading, preprocessing, and tokenization for the LEDGAR dataset.
- [x] **2. Fine-tune a GPT-2 model to tag legal clauses using the classification as generation approach.**  
      Hint: Apply the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill's technique for training your model.
- [x] **3. Evaluate the tagging accuracy of the fine-tuned model.**  
      Hint: Calculate standard classification metrics like accuracy, precision, recall, and F1-score for evaluation.
- [x] **4. Apply dynamic and static post-training quantization to the trained model.**  
      Hint: Use the 'Shrink a Neural Network with Quantisation' drill's methods for applying dynamic and static quantization.
- [ ] **5. Compare the original and quantized model sizes and measure simulated inference times.**  
      Hint: Implement functions to calculate model memory footprint and simulate inference speed for both original and quantized models.
- [ ] **6. Present a dashboard-like output of model efficiency metrics.**  
      Hint: Organize and visualize the collected metrics (accuracy, size, speed) in a clear, comparative dashboard format.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Analyze the impact of quantization on the model's performance for specific legal clause types, identifying categories where accuracy degradation is most or least significant.
- Implement a more detailed breakdown of the simulated inference time, perhaps per model layer or operation, to pinpoint performance bottlenecks.
- Explore different bit-widths for quantization (if supported by the tools used in the drill) and compare their trade-offs in size, speed, and accuracy.
