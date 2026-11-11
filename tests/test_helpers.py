from pathlib import Path

import numpy as np

from quantization_reference import make_onnx_inputs, model_size_mb, onnx_model_path


class _Input:
    def __init__(self, name: str):
        self.name = name


class _Session:
    def get_inputs(self):
        return [_Input("input_ids"), _Input("attention_mask")]


class _Tokenizer:
    def __call__(self, text: str, return_tensors: str):
        assert text == "sample sentence"
        assert return_tensors == "np"
        return {
            "input_ids": np.array([[101, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
            "token_type_ids": np.array([[0, 0]], dtype=np.int64),
        }


def test_model_size_mb(tmp_path: Path):
    artifact = tmp_path / "tiny.bin"
    artifact.write_bytes(b"x" * (1024 * 1024))
    assert model_size_mb(artifact) == 1.0


def test_onnx_model_path_joins_directory_and_filename():
    assert onnx_model_path("models/fp32", "model.onnx") == "models/fp32/model.onnx"


def test_make_onnx_inputs_filters_to_session_signature():
    inputs = make_onnx_inputs(_Session(), _Tokenizer(), "sample sentence")
    assert set(inputs) == {"input_ids", "attention_mask"}
