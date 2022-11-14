import pandas as pd
from joblib import load
from fastapi import FastAPI
import tritonclient.grpc
from src.inference_utils import get_product_nextweek_features, postprocess_prediction
from src.constants import MODEL_JOBLIB_CHECKPOINT, TRITON_ENDPOINT, TRITON_INPUT_NAME, TRITON_OUTPUT_NAME, TRITON_MODEL_NAME

app = FastAPI()
model = load(MODEL_JOBLIB_CHECKPOINT)
weekly_sales_data = pd.read_csv(
    "data/weekly_sales_data.csv"
)  # this is pulled from the database or feature store


triton_client = tritonclient.grpc.InferenceServerClient(
            url=TRITON_ENDPOINT, verbose=False
        )


@app.get("/predict/{product_id}")
async def predict_product_sale(product_id: int):
    next_week, features = get_product_nextweek_features(product_id, weekly_sales_data)

    prediction = model.predict(features)

    prediction = postprocess_prediction(prediction)
    
    prediction = prediction[0]

    result = {"week start": str(next_week.date()), "sales": prediction}

    return result


@app.get("/predict_alt/{product_id}")
async def predict_product_sale_alt(product_id: int):
    next_week, features = get_product_nextweek_features(product_id, weekly_sales_data)

    prediction = get_triton_model_output(features)

    prediction = postprocess_prediction(prediction)

    prediction = prediction.tolist()[0][0]

    result = {"week start": str(next_week.date()), "sales": prediction}

    return result


def get_triton_model_output(features):
    features = features.astype('float32')
    triton_input = tritonclient.grpc.InferInput(TRITON_INPUT_NAME, features.shape, "FP32")
    triton_input.set_data_from_numpy(features)

    output = tritonclient.grpc.InferRequestedOutput(TRITON_OUTPUT_NAME)

    response = triton_client.infer(
            TRITON_MODEL_NAME, model_version="1", inputs=[triton_input], outputs=[output]
        )

    result = response.as_numpy(TRITON_OUTPUT_NAME)

    return result