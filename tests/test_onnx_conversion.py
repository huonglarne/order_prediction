import pickle
from pathlib import Path
from onnxruntime import InferenceSession
import numpy as np
from src.onnx_conversion import sklearn_to_onnx


def test_sklearn_to_onnx():
    model = pickle.load(open("tests/test_data/model.pkl", "rb"))
    example_input = np.array([[4.0, -6.0]])

    expected_result = np.expm1(model.predict(example_input))
    save_path = Path("tests/test_data/model.onnx")
    sklearn_to_onnx(model, example_input, save_path)

    sess = InferenceSession(str(save_path), None)
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: example_input.astype(np.float32)})
    result = result[0][:, 0]

    assert np.allclose(expected_result, result)
    save_path.unlink()
