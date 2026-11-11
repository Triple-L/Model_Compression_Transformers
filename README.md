# Transformer Quantization Lab

A small, reusable learning project for exporting a Transformer classifier to ONNX, applying post-training quantization (PTQ), benchmarking inference latency, comparing predictions, and exploring a separate PyTorch quantization-aware training (QAT) workflow.

> **Scope:** This repository is a practical reference/demo, not a production-ready model-serving system. The QAT routine is intentionally tiny and illustrative; it does not produce a production-quality fine-tuned model.

[中文简介](#中文简介)

## What it demonstrates

- Export a Hugging Face sequence-classification checkpoint to an FP32 ONNX model.
- Apply **dynamic INT8 PTQ** with Optimum and ONNX Runtime. Dynamic PTQ does not require a calibration dataset.
- Optionally apply **static INT8 PTQ**, including representative calibration data and activation-range collection.
- Benchmark FP32 and INT8 ONNX inference with warm-up, per-call measurements, and p50/p95 latency.
- Compare predicted labels on example sentences.
- Optionally inspect a separate, native PyTorch QAT flow. It is not part of the ONNX PTQ benchmark.

## Project layout

```text
.
├── quantization_reference.py   # Single-file CLI and reusable pipeline functions
├── README.md
├── pyproject.toml               # Project metadata and dependencies
├── docs/
│   └── QUANTIZATION_NOTES.md    # Workflow concepts and interpretation guide
├── tests/
│   └── test_helpers.py          # Lightweight helper tests
└── .gitignore                   # Keeps model binaries and course assets out of Git
```

Generated models, checkpoints, downloaded caches, and the original course scaffold/solution notebooks are intentionally excluded from version control. Run the scripts to create artifacts locally.

## Requirements

- Python 3.11 or newer
- A supported CPU architecture (Apple Silicon/ARM64 or x86 with AVX2 for the presets in this example)
- Internet access on first run to download the model and, for static PTQ, the GLUE/SST-2 calibration data

## Setup

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For Conda, Python 3.11 is recommended. Install PyTorch using the command appropriate for your OS and hardware if the default dependency resolver does not select a suitable build.

## Run

By default, export the checkpoint and create a dynamically quantized ONNX model:

```bash
python quantization_reference.py
```

Run the benchmark and compare labels on the included sample sentences:

```bash
python quantization_reference.py --benchmark
```

Also run static PTQ with a shuffled subset of GLUE/SST-2 training examples:

```bash
python quantization_reference.py --static --calibration-samples 50
```

Run the short, **illustrative only** PyTorch QAT loop:

```bash
python quantization_reference.py --qat --qat-steps 3
```

Run all optional demonstrations:

```bash
python quantization_reference.py --benchmark --static --qat
```

The script creates `onnx_fp32/`, `onnx_dynamic_int8/`, and optionally `onnx_static_int8/` in the current working directory. QAT writes a PyTorch state dictionary. These generated artifacts are ignored by Git.

## Interpreting results

The benchmark reports p50 and p95 in milliseconds. Improvement is computed as:

```text
(FP32 latency - INT8 latency) / FP32 latency × 100%
```

A positive value means INT8 was faster in that run; a negative value means it was slower. Results depend on processor, runtime build, thread settings, input length, and system load. Use repeated runs and representative inputs before drawing deployment conclusions.

A previous local run observed a model-size reduction of about 74.9%, with p50 and p95 improvements around 35.69% and 42.91%. Treat these only as example measurements—not expected or guaranteed results.

Matching labels on three sample sentences is a smoke test, not an accuracy evaluation. Validate quantized models on a sufficiently large, representative held-out dataset and compare task metrics before deployment.

## Important distinctions

- **Dynamic PTQ:** post-training quantization; no calibration dataset is used by this workflow.
- **Static PTQ:** requires representative calibration data to estimate activation ranges.
- **QAT:** inserts fake-quantization behavior during training and follows a separate PyTorch path. It does not automatically produce an ONNX model. The short demo here is for illustrating mechanics only.

See [docs/QUANTIZATION_NOTES.md](docs/QUANTIZATION_NOTES.md) for a concise workflow and decision guide.

## Tests

```bash
pytest
```

## Learning / attribution note

This is an independently organized reference project based on publicly available Python APIs and the author's local experiments. The course-provided scaffold and solution notebooks are not redistributed here. Check the relevant library and dataset terms before using this code or downloaded artifacts in another setting.

## 中文简介

这是一个可复用的 Transformer 量化学习项目，演示 ONNX 导出、动态/静态 PTQ、p50/p95 推理延迟测试、预测标签比较，以及独立的 PyTorch QAT 入门流程。QAT 示例只有少量训练步骤，仅用于理解流程，不适合作为真实微调结果。模型文件、检查点和课程原始 notebook 不会提交到仓库；运行脚本后可在本地生成模型产物。
