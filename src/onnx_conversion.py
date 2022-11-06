from typing import Any
from pathlib import Path
from numpy import ndarray
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn


def sklearn_to_onnx(model: Any, example_input: ndarray, save_path: Path):
    n_features = example_input[0].shape
    input_shape = (None,) + n_features

    initial_type = [("float_input", FloatTensorType(input_shape))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(save_path, "wb") as f:
        f.write(onx.SerializeToString())
