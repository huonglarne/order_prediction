from typing import Any
from pathlib import Path
from numpy import ndarray
import numpy as np
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from onnxruntime import InferenceSession


def sklearn_to_onnx(model: Any, example_input: ndarray, save_path: Path):
    """Convert a sklearn model to ONNX format and save it to disk.

    # Args:
    #     model (Any): A sklearn model.
    #     example_input (ndarray): Example input to the model, small batchsize.
    #     save_path (Path): Path to save the ONNX model to.
    """
    n_features = example_input[0].shape
    input_shape = (None,) + n_features

    initial_type = [("float_input", FloatTensorType(input_shape))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(save_path, "wb") as f:
        f.write(onx.SerializeToString())


def get_onnx_session(model_path: Path):
    """Load an ONNX model from disk."""
    return InferenceSession(str(model_path), None)


def onnx_infere_single_inp_outp(session, input_data: ndarray):
    """Run inference on an ONNX model with a single input and output."""
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data.astype(np.float32)})
    return result[0]
