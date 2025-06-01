#!/bin/bash


# change this to whatever we want to publish our messages to
export PREDICTION_ENDPOINT="https://webhook.site/0bc39358-0532-4bb5-912e-9d41856a6afd"
export USINGQUEUE=True
export QUEUEHOST="localhost"

python -u server.py
