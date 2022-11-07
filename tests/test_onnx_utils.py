import pickle
from pathlib import Path
import numpy as np
from src.onnx_utils import get_onnx_session, onnx_inference_single_input, sklearn_to_onnx


def test_sklearn_to_onnx():
    """Docstring"""
    model = pickle.load(open("tests/test_data/model.pkl", "rb"))
    example_input = np.array([[4.0, -6.0]])

    expected_result = model.predict(example_input)
    save_path = Path("tests/test_data/model.onnx")
    sklearn_to_onnx(model, example_input, save_path)

    session = get_onnx_session(save_path)
    result = onnx_inference_single_input(session, example_input)
    result = result[:, 0]

    assert np.allclose(expected_result, result)
    save_path.unlink()
