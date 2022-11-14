# Order prediction

# Overview

## Objective

- Make predictions on the quantity of orders of a given product in the next days.

- Create a backend service to serve the model in a production environment.

## Solution

- Train a Random Forest Regressor with weekly sales data to predict the sales of any product in future weeks.

- Serve the prediction model in a Docker container with an API endpoint exposed to the outside.

# Set up

## Development environment

All dependencies have been installed in the [VS Code dev container](https://code.visualstudio.com/docs/devcontainers/containers).

To open:

- Open the repository in VS Code

- Open command palette: ```Ctrl + Shift + P ```

- Type "Open folder in container" and select

## Production environment

Docker and Docker Compose should be installed on your machine in order to run the production container.

# Training

## Run

Inside the VS Code dev container, run this command:

```
python train.py
```

## Explain

### **Feature engineering**

- From the [order table](data/data_order.csv) in the database, calculate the [weekly ordered quantity](data/weekly_sales_data.csv) of each product.

    *Note*: My code lets us specify data from a particular time period because:
    - too-far-in-the-past data might be outdated and don't represent the current situation well.
    - we might want to retrain or fine-tune the model with most current data.

- Use sales data and sales difference from previous weeks as [features](data/features.csv) to predict a week's sales.

    A feature of a product in a particular week looks like this:

    |last-1_week_sales|last-1_week_diff|last-2_week_sales|last-2_week_diff|   |
    |---|---|---|---|---|
    |4.0|-25.0|29.0|19.0|

### **Model**

- Train a Random forest regressor with Root mean square error as the loss function.

- Checkpoint the model as SKlearn joblib and Onnx.
    
    Why I chose these 2 methods will be explained in the next section.

# Serving

## Inference

## Simple serving method

Due to the small size of the model and the simple use case, we can simple put the model inside a container and infere with SKlearn's default run-time through a simple FastAPI endpoint.

How to run:

Build and run the containerized model:

```
docker build -f ./docker/Dockerfile app:latest --target production .

docker run --network=host app:latest
```

Predict the next-week sales of product with ID 4048:

```
curl 0.0.0.0:9001/predict/4048
```

## More sophisticated solution

This solution might be overkill for this use case but it's more practical if we want to deploy a bigger model and serve it more dynamically serving in real life.

- We separate data pre/post-processing and model inference into 2 different endpoints.

    If we deploy this using AWS services, the simple architecture should look like this:

    ```
    API Gateway (receive requests from users) <--> Lambda function (pre/post-processing) <--> ECS/Fargate (model inference)
    ```

    The advantage of this deployment architecture is that the service in each endpoint can auto-scale based on traffic. This also comes with economic advantage.

    Separating model inference with other operations also means if we want to retrain the model due to data drift, we only need to redeploy the container with the retrained model, the other services can stay the same.

- We can also serve the model with Triton server, which supports GPU run-time and dynamic batching. We convert the SKlearn model to Onnx so that it can run with Triton backend.

    Triton only supports model inference so data pre/post-processing must happen somewhere else, for example a separate Lambda function.


We can simulate this pipeline locally using 2 containers "app" and "triton":

```
"app" : FastAPI gateway (receive requests from users + pre/post-processing) <--> "triton": Triton server (model inference)
```

How to run:

```
docker compose -f docker/docker-compose.yml up
```

The IP of the "app" container is usually 172.21.0.2 but if it's different, you can use ```docker inspect``` to find out its IP on your machine.

Predict the next-week sales of product with ID 4048:

```
curl 172.21.0.2:9001/predict_alt/4048
```


# Further improvements

These are the things I didn't have time to do but would be nice-to-have addition:

- Predict the sales for not only the next week but many weeks after that.

- Use latest features, pulled from a constantly updating database or feature store to predict next weeks' sales, instead of using static csv files inside the serving container.

- CI/CD pipeline, scheduling for model retraining in case of new incoming data.