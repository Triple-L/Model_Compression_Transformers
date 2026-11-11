# Quantization workflow notes

## End-to-end picture

```text
Hugging Face checkpoint
        │
        ├── tokenizer ── text → input_ids / attention_mask
        │
        └── export=True → FP32 ONNX → ORTQuantizer → INT8 ONNX
                                                │
                         ONNX Runtime sessions ─┴─ benchmark / compare labels
```

The PyTorch QAT example is a separate branch:

```text
PyTorch checkpoint → assign qconfig → prepare_qat → short training loop → convert
```

That native PyTorch branch does not export ONNX in this project and is not included in the ONNX Runtime benchmark.

## PTQ choices

| Method | Calibration data | Main idea | Implemented here |
|---|---:|---|---|
| Dynamic PTQ | No | Quantize supported weights; activation handling occurs at inference | Yes; default path |
| Static PTQ | Yes | Collect representative activation ranges before quantizing | Optional `--static` path |
| QAT | Training data | Simulate quantization effects during training | Tiny educational PyTorch demo only |

Calibration examples should resemble real model inputs in language, length, and formatting. The example uses GLUE/SST-2 training sentences because the checkpoint is an English SST-2 classifier. Labels are not needed for range calibration. Keep calibration, validation, and test data conceptually separate.

## Benchmark interpretation

- Warm up the runtime before measuring to avoid including startup effects in steady-state latency.
- Record each inference duration individually; aggregate total time cannot yield p50/p95.
- Use identical text and preprocessing for the FP32 and INT8 sessions. Filter inputs to each session's declared names.
- p50 is the median; p95 is the latency below which 95% of observed calls fall.
- Report environment and repeat measurements; CPU scheduling, runtime threading, and thermal state can shift results.

## Deployment decision checklist

1. Does INT8 meet the explicit p95 service-level objective on deployment hardware?
2. Does a representative held-out evaluation show acceptable task metrics, not just matching three sample predictions?
3. If dynamic PTQ is insufficient, test static PTQ with representative calibration inputs.
4. Consider QAT only when the accuracy/performance trade-off warrants the added training and deployment complexity.
