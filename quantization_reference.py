"""Practical reference: Transformer quantization with Optimum/ONNX Runtime and PyTorch QAT.

Environment (Python 3.11):
    pip install transformers "optimum-onnx[onnxruntime]" datasets onnxruntime numpy torch

Examples:
    python quantization_reference.py                  # export + dynamic PTQ
    python quantization_reference.py --benchmark       # also benchmark and compare labels
    python quantization_reference.py --static          # also run static PTQ (downloads GLUE/SST-2)
    python quantization_reference.py --qat              # also run a tiny illustrative PyTorch QAT loop

Dynamic/static PTQ here produce ONNX files. The QAT example is a separate native PyTorch
workflow and saves a state_dict; it is deliberately not included in the ONNX benchmark.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from functools import partial
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.ao.quantization as tq
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import (
    AutoCalibrationConfig,
    AutoQuantizationConfig,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# -----------------------------------------------------------------------------
# Shared configuration and helpers
# -----------------------------------------------------------------------------
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
FP32_DIR = "onnx_fp32"
DYNAMIC_DIR = "onnx_dynamic_int8"
STATIC_DIR = "onnx_static_int8"
QAT_CHECKPOINT = "distilbert_qat_int8.pt"
SAMPLE_TEXTS = [
    "The product arrived on time and works exactly as described.",
    "Customer support was unhelpful and the issue was never resolved.",
    "Shipping was fast but the packaging was damaged.",
]


def detect_hardware() -> tuple[str, str]:
    """Return Optimum's ONNX preset and PyTorch's quantized CPU engine."""
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return "arm64", "qnnpack"
    return "avx2", "fbgemm"


def model_size_mb(path: str | Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def onnx_model_path(directory: str | Path, filename: str) -> str:
    return os.path.join(directory, filename)


def print_size_comparison(fp32_path: str, quantized_path: str) -> None:
    fp32_mb = model_size_mb(fp32_path)
    quantized_mb = model_size_mb(quantized_path)
    reduction = (1 - quantized_mb / fp32_mb) * 100
    print(f"FP32 ONNX: {fp32_mb:.1f} MB")
    print(f"Quantized ONNX: {quantized_mb:.1f} MB")
    print(f"Size reduction: {reduction:.1f}%")


# -----------------------------------------------------------------------------
# 1. Export a Transformers checkpoint to a reusable FP32 ONNX baseline
# -----------------------------------------------------------------------------
def export_fp32_onnx(model_id: str = MODEL_ID, output_dir: str = FP32_DIR):
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    return model


# -----------------------------------------------------------------------------
# 2. Post-training quantization (PTQ) using Optimum's ORTQuantizer
# -----------------------------------------------------------------------------
def run_dynamic_ptq(
    onnx_model,
    output_dir: str = DYNAMIC_DIR,
    onnx_arch: str | None = None,
) -> str:
    if onnx_arch is None:
        onnx_arch, _ = detect_hardware()
    quantizer = ORTQuantizer.from_pretrained(onnx_model)
    config = getattr(AutoQuantizationConfig, onnx_arch)(
        is_static=False,
        per_channel=False,
    )
    quantizer.quantize(save_dir=output_dir, quantization_config=config)
    return onnx_model_path(output_dir, "model_quantized.onnx")


def preprocess_calibration_batch(examples, tokenizer):
    """Convert SST-2 sentence examples into the ONNX model's text inputs."""
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def run_static_ptq(
    onnx_model,
    tokenizer,
    output_dir: str = STATIC_DIR,
    onnx_arch: str | None = None,
    calibration_samples: int = 50,
) -> str:
    """Run calibration first, then static quantization; calibration data is required."""
    if onnx_arch is None:
        onnx_arch, _ = detect_hardware()
    quantizer = ORTQuantizer.from_pretrained(onnx_model)
    config = getattr(AutoQuantizationConfig, onnx_arch)(
        is_static=True,
        per_channel=False,
    )
    dataset = quantizer.get_calibration_dataset(
        "glue",
        dataset_config_name="sst2",
        dataset_split="train",
        num_samples=calibration_samples,
        preprocess_function=partial(preprocess_calibration_batch, tokenizer=tokenizer),
    )
    calibration_config = AutoCalibrationConfig.minmax(dataset)
    ranges = quantizer.fit(
        dataset=dataset,
        calibration_config=calibration_config,
        operators_to_quantize=config.operators_to_quantize,
    )
    quantizer.quantize(
        save_dir=output_dir,
        calibration_tensors_range=ranges,
        quantization_config=config,
    )
    return onnx_model_path(output_dir, "model_quantized.onnx")


# -----------------------------------------------------------------------------
# 3. PyTorch-native QAT demonstration (independent of the ONNX PTQ path)
# -----------------------------------------------------------------------------
def run_qat_demo(
    model_id: str = MODEL_ID,
    output_path: str = QAT_CHECKPOINT,
    steps: int = 3,
) -> str:
    """Illustrate QAT mechanics; a few toy steps are not meaningful fine-tuning."""
    _, quant_engine = detect_hardware()
    torch.backends.quantized.engine = quant_engine
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.train()
    model.qconfig = tq.get_default_qat_qconfig(quant_engine)

    # Embedding layers need a compatible embedding-specific recipe; skip the
    # generic QAT qconfig here rather than applying it to integer token indices.
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = None

    tq.prepare_qat(model, inplace=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    texts = ["a fun and charming film", "a tedious and disappointing movie"]
    labels = torch.tensor([1, 0])
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(**batch, labels=labels)
        outputs.loss.backward()
        optimizer.step()

    model.eval()
    quantized_model = tq.convert(model, inplace=False)
    torch.save(quantized_model.state_dict(), output_path)
    print(f"Illustrative QAT loss: {outputs.loss.item():.4f}")
    print(f"Saved QAT state_dict: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# 4. ONNX inference inputs, latency measurement, and prediction comparison
# -----------------------------------------------------------------------------
def make_onnx_inputs(session, tokenizer, text: str) -> dict[str, np.ndarray]:
    encoded = tokenizer(text, return_tensors="np")
    expected_names = {item.name for item in session.get_inputs()}
    missing = expected_names - set(encoded.keys())
    if missing:
        raise ValueError(f"Tokenizer did not produce required ONNX inputs: {sorted(missing)}")
    return {name: encoded[name] for name in expected_names}


def benchmark_model(
    session,
    tokenized_inputs: dict[str, np.ndarray],
    warmup_iters: int = 10,
    timed_iters: int = 50,
) -> tuple[float, float]:
    """Return per-call p50/p95 latency in milliseconds after untimed warm-up."""
    if warmup_iters < 10:
        raise ValueError("Use at least 10 warm-up iterations for this activity.")
    if timed_iters < 1:
        raise ValueError("timed_iters must be at least 1.")

    for _ in range(warmup_iters):
        session.run(None, tokenized_inputs)

    latencies_ms = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        session.run(None, tokenized_inputs)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    p50, p95 = np.percentile(latencies_ms, [50, 95])
    return float(p50), float(p95)


def benchmark_onnx_pair(
    fp32_dir: str = FP32_DIR,
    quantized_dir: str = DYNAMIC_DIR,
    model_id: str = MODEL_ID,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fp32_session = ort.InferenceSession(onnx_model_path(fp32_dir, "model.onnx"))
    int8_session = ort.InferenceSession(
        onnx_model_path(quantized_dir, "model_quantized.onnx")
    )
    sample = SAMPLE_TEXTS[0]
    fp32_inputs = make_onnx_inputs(fp32_session, tokenizer, sample)
    int8_inputs = make_onnx_inputs(int8_session, tokenizer, sample)

    fp32_p50, fp32_p95 = benchmark_model(fp32_session, fp32_inputs)
    int8_p50, int8_p95 = benchmark_model(int8_session, int8_inputs)
    print(f"FP32 p50: {fp32_p50:.2f} ms | p95: {fp32_p95:.2f} ms")
    print(f"INT8 p50: {int8_p50:.2f} ms | p95: {int8_p95:.2f} ms")
    print(f"p50 improvement: {(fp32_p50 - int8_p50) / fp32_p50 * 100:.2f}%")
    print(f"p95 improvement: {(fp32_p95 - int8_p95) / fp32_p95 * 100:.2f}%")

    print("Label comparison on sample texts:")
    for text in SAMPLE_TEXTS:
        fp32_inputs = make_onnx_inputs(fp32_session, tokenizer, text)
        int8_inputs = make_onnx_inputs(int8_session, tokenizer, text)
        fp32_logits = fp32_session.run(None, fp32_inputs)[0]
        int8_logits = int8_session.run(None, int8_inputs)[0]
        fp32_label = int(np.argmax(fp32_logits, axis=-1)[0])
        int8_label = int(np.argmax(int8_logits, axis=-1)[0])
        result = "match" if fp32_label == int8_label else "MISMATCH"
        print(f"[{result}] FP32={fp32_label} INT8={int8_label} | {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true", help="also run static PTQ")
    parser.add_argument("--qat", action="store_true", help="also run the illustrative PyTorch QAT demo")
    parser.add_argument("--benchmark", action="store_true", help="benchmark dynamic ONNX PTQ and compare labels")
    parser.add_argument("--calibration-samples", type=int, default=50)
    parser.add_argument("--qat-steps", type=int, default=3)
    args = parser.parse_args()

    onnx_arch, _ = detect_hardware()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    onnx_model = export_fp32_onnx()
    dynamic_path = run_dynamic_ptq(onnx_model, onnx_arch=onnx_arch)
    fp32_path = onnx_model_path(FP32_DIR, "model.onnx")
    print_size_comparison(fp32_path, dynamic_path)

    if args.benchmark:
        benchmark_onnx_pair()

    if args.static:
        static_path = run_static_ptq(
            onnx_model,
            tokenizer,
            onnx_arch=onnx_arch,
            calibration_samples=args.calibration_samples,
        )
        print_size_comparison(fp32_path, static_path)

    if args.qat:
        run_qat_demo(steps=args.qat_steps)


if __name__ == "__main__":
    main()
