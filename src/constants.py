RELEVANT_COLUMNS = {"PRODUCT_ID", "QUANTITY", "CHECKOUT_DATE"}
NON_FEATURES = {"sales", "product", "week_start"}
NUM_PAST_WEEKS = 2

MODEL_JOBLIB_CHECKPOINT = "checkpoints/model.joblib"
MODEL_ONNX_CHECKPOINT = "models/prediction_model/1/model.onnx"

TRITON_ENDPOINT = "triton:8001"
TRITON_MODEL_NAME = "prediction_model"
TRITON_INPUT_NAME = "float_input"
TRITON_OUTPUT_NAME = "variable"