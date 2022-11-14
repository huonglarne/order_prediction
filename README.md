# Order prediction

# Objective

- Make predictions on the quantity of orders of a given product in the next days.

- Create a backend service to serve the model in a 'production' environment.

# Solution

- Train a Random Forest Regressor with weekly sales data to predict the sales of any product in future weeks.

- Serve the prediction model in a Docker container with an API endpoint exposed to the outside.

# Set up

## Development environment

All dependencies have been installed in the [VS Code dev container](https://code.visualstudio.com/docs/devcontainers/containers).

This environment is used for data processing, training and other development activities.

To open:

- Open the repository in VS Code

- Open command palette: ```Ctrl + P ``` and then ```>```

- Type "Open folder in container" and select

## Production environment

Docker should be installed on your machine in order to run the production container.

# Training

Run inside VS Code dev container:

```
python train.py
```


# Serving

## Serving from local machine

Build the image:

```
docker build order_pred:latest --target production .
```

Open a local host from the container to serve the model:

```
docker run --net-work=host order_pred:latest
```

Predict the sales for next week for product with ID 4048:

```
curl app:9001/predict/4048
```


```
uvicorn src.prediction_api:app --host 0.0.0.0 --port 80
```

```
docker build -t name:tag --build-arg MODEL_BASE_DIR=models -f .devcontainer/server/Dockerfile .
```

```
docker compose -f docker/docker-compose.yml build
```

```
curl 172.21.0.2:9001/predict_alt/4048
```