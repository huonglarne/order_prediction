# Order prediction

## Objective

- Make predictions in terms of the quantity of a given product in the next 30 days.

- Create a backend service to serve the model in a 'production' environment.

## Solution

# How to run

Train

```
python train.py
```

Infere

```
python serve.py
```

```
curl http://0.0.0.0:8000/predict/4048
```