RELEVANT_COLUMNS = {"PRODUCT_ID", "QUANTITY", "CHECKOUT_DATE"}
NON_FEATURES = {"sales", "product", "week_start"}
NUM_PAST_WEEKS = 2

MODEL_JOBLIB_CHECKPOINT = "checkpoints/model.joblib"
MODEL_ONNX_CHECKPOINT = "models/prediction_model/1/model.onnx"
