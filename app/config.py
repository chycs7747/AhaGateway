# AhaGateway settings — edit these to match your own vLLM setup.
#
# The defaults below are the author's local environment (a Qwen3 container
# named "qwen38" serving on port 8078). Nothing here is secret; replace the
# values and restart the server.

# Base URL of your vLLM OpenAI-compatible API (e.g. "http://<host>:<port>/v1")
VLLM_BASE_URL = "http://localhost:8078/v1"

# Served model name — must match vLLM's --served-model-name
# (check with: curl <VLLM_BASE_URL>/models)
VLLM_MODEL = "qwen38-nvfp4"

# Name of the Docker container running vLLM (docker ps --format '{{.Names}}')
VLLM_CONTAINER = "qwen38"

VLLM_TIMEOUT = 120.0        # total wait for a completion (seconds)
VLLM_CONNECT_TIMEOUT = 5.0  # connection establishment limit (seconds)
