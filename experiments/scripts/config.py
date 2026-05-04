"""
Experiment configuration — API keys and model settings.

Loads .env from the project root by default; values can also be overridden here directly.
"""

import os
from pathlib import Path

# ---------- load .env ----------
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

def _load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

_load_env(_ENV_PATH)

# ---------- LLM (used by the agent to generate responses) ----------
LLM_API_KEY = os.getenv("API_KEY", "")
LLM_API_BASE = os.getenv("API_BASE", "https://api.qingyuntop.top/v1")
LLM_MODEL = os.getenv("MODEL_NAME", "gpt-4.1-nano")  # cheapest by default
LLM_TIMEOUT = int(os.getenv("TIMEOUT", "60"))

# ---------- Embedding (used by the baseline and some memory backends) ----------
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://aihubmix.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4b")

# ---------- Mem0-specific ----------
MEM0_LLM_CONFIG = {
    "provider": "openai",
    "config": {
        "model": os.getenv("MEM0_MODEL", LLM_MODEL),
        "api_key": LLM_API_KEY,
        # openai_base_url routes Mem0 through our proxy
        "openai_base_url": LLM_API_BASE,
    },
}

MEM0_EMBEDDER_CONFIG = {
    "provider": "openai",
    "config": {
        "model": "text-embedding-3-small",
        "api_key": EMBEDDING_API_KEY,
        "openai_base_url": EMBEDDING_API_BASE,
        "embedding_dims": 1536,
    },
}

# ---------- data paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmark"
CHARACTERS_PATH = DATA_DIR / "characters.json"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------- experiment parameters ----------
# First round: only run CHAR_01, one scenario per stage
TARGET_CHAR_ID = "CHAR_01"
TOP_K = 5  # number of retrieved memories
