# AhaGateway settings — template.
#
# Copy this file to app/config.py and fill in your own values.
# app/config.py is git-ignored so your local paths and names stay private.

# Base URL of the vLLM OpenAI-compatible API. All model containers share
# this port: the GPU fits one model at a time, so whichever container is
# running owns the port. The manager enforces that exclusivity.
VLLM_BASE_URL = "http://localhost:8000/v1"

VLLM_TIMEOUT = 120.0        # total wait for a completion (seconds)
VLLM_CONNECT_TIMEOUT = 5.0  # connection establishment limit (seconds)

# How long ensure() waits for a model to finish loading (seconds).
MODEL_READY_TIMEOUT = 600.0

VLLM_IMAGE = "vllm/vllm-openai:latest"

# Model registry: gateway-facing name -> how to run/reach it.
#   container:   docker container name
#   served_name: vLLM --served-model-name (the "model" field sent to vLLM)
#   spec:        how to CREATE the container if it does not exist.
#                None = the container is managed outside the gateway
#                (pre-created by hand), we only start/stop it.
MODELS = {
    "your-model": {
        "container": "your-model-container",
        "served_name": "your-model",
        "spec": {
            "image": VLLM_IMAGE,
            "model_dir": "/path/to/your/models",  # mounted at /models
            "args": [
                "vllm", "serve", "/models/Your-Model-Name",
                "--served-model-name", "your-model",
                "--host", "0.0.0.0", "--port", "8000",
                "--max-model-len", "32768",
                "--gpu-memory-utilization", "0.80",
            ],
        },
    },
}
