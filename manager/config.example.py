import os
# Manager settings — template. Copy to manager/config.py and fill in your values.
# manager/config.py is git-ignored so your local paths and names stay private.

# The vLLM OpenAI-compatible API on THIS machine. All model containers share
# this port: the GPU fits one model at a time; the manager enforces that.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

# ---- time constants: each one measures a different thing ----

# [loading] how long a switch waits for the new model to become ready.
# Raise this for bigger models (8-10 min load times).
MODEL_READY_TIMEOUT = 600.0

# [drain] how long a switch/unload waits for in-flight inference to finish.
# Unrelated to model size — it measures other requests' generation time.
# Keep it larger than the gateway's per-request VLLM_TIMEOUT.
SWITCH_DRAIN_TIMEOUT = 180.0

# [ghost sessions] unreleased sessions (dead clients) expire after this.
SESSION_TTL = 1800.0

VLLM_IMAGE = "vllm/vllm-openai:latest"

# Model registry: gateway-facing name -> how to run/reach it.
#   container:   docker container name
#   served_name: vLLM --served-model-name (the "model" field sent to vLLM)
#   spec:        how to CREATE the container if it does not exist.
#                None = pre-created by hand; we only start/stop it.
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
