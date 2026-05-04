"""Mem0 quickstart chat — OpenAI-compatible LLM gateway + qwen3-embedding-4b (aihubmix).

Usage:
    source .venv/bin/activate
    cp .env.example .env  # then edit .env to fill in API_KEY, API_BASE, AIHUBMIX_API_KEY
    python chat.py
"""

import os
import sys
from pathlib import Path

# ---- Load .env (no extra dependency) ---------------------------------------
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v or v.startswith("sk-your-"):
        raise SystemExit(
            f"{name} is not set. Edit .env (copy from .env.example) and try again."
        )
    return v


LLM_KEY = _require("API_KEY")
LLM_BASE = os.environ.get("API_BASE", "")
if not LLM_BASE:
    sys.exit("API_BASE is not set in .env")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.4-mini")

EMBED_KEY = _require("AIHUBMIX_API_KEY")
EMBED_BASE = os.environ.get("AIHUBMIX_API_BASE", "https://aihubmix.com/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding-4b")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "2560"))

from openai import OpenAI
from mem0 import Memory

# ---- Mem0 config -----------------------------------------------------------
# LLM and embedder live on different proxies, so we pass openai_base_url +
# api_key to each. The vector store dim must match the embedder (Qwen3-4B = 2560);
# we use a fresh collection name so it doesn't collide with any prior 1536-dim store.
mem0_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": LLM_MODEL,
            "openai_base_url": LLM_BASE,
            "api_key": LLM_KEY,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": EMBED_MODEL,
            "openai_base_url": EMBED_BASE,
            "api_key": EMBED_KEY,
            # NOTE: do NOT set embedding_dims here. mem0's OpenAIEmbedding would
            # then send a `dimensions=` field, which the aihubmix Qwen3 backend
            # rejects (400 invalid_argument). Qwen3-Embedding-4B is fixed at 2560
            # natively; the vector_store dim below must match it.
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_qwen3_4b",
            "embedding_model_dims": EMBED_DIMS,
            "path": str(Path(__file__).parent / ".mem0" / "qdrant"),
        },
    },
}

memory = Memory.from_config(mem0_config)

# Chat client points at the SJTU proxy too.
chat_client = OpenAI(api_key=LLM_KEY, base_url=LLM_BASE)


def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    relevant_memories = memory.search(
        query=message, filters={"user_id": user_id}, top_k=3
    )
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
    )

    system_prompt = (
        "You are a helpful AI. Answer the question based on query and memories.\n"
        f"User Memories:\n{memories_str}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = chat_client.chat.completions.create(
        model=LLM_MODEL, messages=messages
    )
    assistant_response = response.choices[0].message.content

    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)
    return assistant_response


def main() -> None:
    print(f"Chat with AI ({LLM_MODEL} + {EMBED_MODEL}). Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")


if __name__ == "__main__":
    main()
