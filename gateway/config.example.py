import os
# Gateway settings — template. Copy to gateway/config.py and fill in your values.
# gateway/config.py is git-ignored.

# Where the manager service runs (internal traffic).
MANAGER_BASE_URL = os.environ.get("MANAGER_BASE_URL", "http://localhost:9100")

# [one inference] how long to wait for a completion — 504 to the client beyond this.
VLLM_TIMEOUT = 120.0
VLLM_CONNECT_TIMEOUT = 5.0

# [session acquire] the manager may drain (SWITCH_DRAIN_TIMEOUT) and load a model
# (MODEL_READY_TIMEOUT) before answering — keep this above their sum.
SESSION_ACQUIRE_TIMEOUT = 800.0
