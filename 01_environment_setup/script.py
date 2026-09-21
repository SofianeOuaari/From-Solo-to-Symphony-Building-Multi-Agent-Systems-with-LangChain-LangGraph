# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: imprs_workshop
#     language: python
#     name: imprs_workshop
# ---

# %% [markdown]
# # 01 · Setting up the environment
#
# **Goal of this notebook:** by the end of it, every red ❌ below is a green ✅.
# Nothing here is about agents yet — we are just making sure the tools work, so
# that later, when something breaks, you know it is *your idea* breaking and not
# your install.
#
# We will check four things:
#
# | # | What | Why we need it |
# |---|------|----------------|
# | 1 | Python + packages | LangChain, LangGraph, numpy, plotting |
# | 2 | Ollama | runs LLMs *on your laptop* — private, free, offline |
# | 3 | The LiteLLM proxy | a shared server with a *bigger* model, needs a key |
# | 4 | `.env` | one file where your model choice lives |
#
# You only strictly need **one** of (2) or (3). Having both is nicer, because
# you can feel the difference between a 2-billion-parameter model on your own
# CPU and a 35-billion-parameter model on a GPU cluster.

# %% [markdown]
# ## 0 · Creating the environment
#
# Do this **once**, in a terminal, *before* opening this notebook. Pick whichever
# of the two you prefer — they achieve exactly the same thing.
#
# ### Option A — conda (familiar, heavier)
#
# ```bash
# cd /path/to/IMPRS_LLM_Workshop
# conda create -y -n imprs_workshop python=3.11
# conda activate imprs_workshop
# pip install -r requirements.txt
#
# # make the environment visible to Jupyter as a kernel
# python -m ipykernel install --user --name imprs_workshop \
#        --display-name "imprs_workshop"
#
# jupyter lab
# ```
#
# ### Option B — uv (same result, ~10× faster)
#
# [uv](https://docs.astral.sh/uv/) is a drop-in replacement for pip/venv.
#
# ```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh    # if you don't have it
#
# cd /path/to/IMPRS_LLM_Workshop
# uv venv --python 3.11 .venv
# source .venv/bin/activate
# uv pip install -r requirements.txt
#
# python -m ipykernel install --user --name imprs_workshop \
#        --display-name "imprs_workshop"
#
# jupyter lab
# ```
#
# Then, **top-right of the notebook, choose the `imprs_workshop` kernel.**
# If the imports below fail, that selection is the first thing to check.

# %% [markdown]
# ### The safety net
#
# Every notebook in this workshop starts with the cell below, so that people who
# jump straight into notebook 07 still get a working environment. If you already
# did step 0 it finishes in a second and changes nothing.

# %%
# %pip install -q -r ../requirements.txt

# %% [markdown]
# ## 1 · Python and the packages

# %%
import sys
import platform

print("python  :", sys.version.split()[0], f"({platform.system()})")
print("executable:", sys.executable)

if sys.version_info < (3, 10):
    print("\n❌ Please use Python 3.10+ — LangChain 1.x needs it.")
else:
    print("\n✅ Python version is fine.")

# %%
# Import everything the workshop needs, and report per package.
# A missing package here is not a disaster -- just re-run the %pip cell above.
checks = {
    "langchain": "langchain",
    "langchain_core": "langchain_core",
    "langchain_ollama": "langchain_ollama",
    "langchain_openai": "langchain_openai",
    "langgraph": "langgraph",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "umap": "umap-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pydantic": "pydantic",
    "dotenv": "python-dotenv",
}

import importlib

missing = []
for module, pip_name in checks.items():
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "?")
        print(f"✅ {pip_name:<16} {version}")
    except Exception as exc:
        missing.append(pip_name)
        print(f"❌ {pip_name:<16} {type(exc).__name__}")

print()
if missing:
    print("Install the missing ones with:\n   pip install " + " ".join(missing))
else:
    print("✅ All packages importable.")

# %% [markdown]
# ## 2 · Ollama — models on your own machine
#
# [Ollama](https://ollama.com) is a small server that downloads open-weight
# models and exposes them on `http://localhost:11434`. Think of it as *Docker
# for language models*: `ollama pull <model>` then `ollama run <model>`.
#
# Install (once, in a terminal):
#
# ```bash
# curl -fsSL https://ollama.com/install.sh | sh    # Linux
# # macOS: download the app from ollama.com
# ollama serve        # usually already running as a service
# ```
#
# Then pull the models this workshop uses:
#
# ```bash
# ollama pull qwen3.5:2b               # our small chat model (notebooks 02+)
# ollama pull deepseek-r1:1.5b         # a "thinking out loud" model (notebook 06)
#
# ollama pull nomic-embed-text:latest  # embeddings (notebooks 03, 04, 05)
# ollama pull qwen3-embedding:0.6b
# ollama pull all-minilm:33m
# ```
#
# Total download ≈ 7 GB. Start it now if you haven't — it downloads while we talk.

# %%
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434"

def ollama_tags(base_url: str = OLLAMA_URL) -> list[str]:
    """Ask the local Ollama server which models it has downloaded."""
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=10) as resp:
        payload = json.load(resp)
    return sorted(m["name"] for m in payload.get("models", []))


try:
    installed = ollama_tags()
    print(f"✅ Ollama is running at {OLLAMA_URL} with {len(installed)} model(s):\n")
    for name in installed:
        print("   ", name)
except Exception as exc:
    installed = []
    print(f"❌ Could not reach Ollama at {OLLAMA_URL}  ({type(exc).__name__})")
    print("   Start it with:  ollama serve")

# %%
# Do we have the specific models the workshop asks for?
NEEDED = [
    ("qwen3.5:2b", "chat model, notebooks 02+"),
    ("deepseek-r1:1.5b", "reasoning model, notebook 06"),
    ("nomic-embed-text:latest", "embeddings, notebooks 03-05"),
    ("qwen3-embedding:0.6b", "embeddings comparison, notebook 03"),
    ("all-minilm:33m", "embeddings comparison, notebook 03"),
]

to_pull = []
for name, purpose in NEEDED:
    # Ollama reports "qwen3.5:2b"; be forgiving about the ":latest" suffix.
    stem = name.split(":")[0]
    have = any(i == name or i.split(":")[0] == stem for i in installed)
    print(f"{'✅' if have else '⬇️ '} {name:<26} {purpose}")
    if not have:
        to_pull.append(name)

if to_pull:
    print("\nRun this in a terminal:\n")
    for name in to_pull:
        print(f"   ollama pull {name}")
    print("\n(Smaller substitutes are fine — e.g. qwen3.5:0.8b instead of 2b.")
    print(" Just change the name in .env; every notebook reads it from there.)")

# %% [markdown]
# ## 3 · The LiteLLM proxy — a bigger model, shared
#
# A **LiteLLM proxy** sits in front of one or more model servers and speaks the
# OpenAI HTTP API. That matters for us for one reason:
#
# > any client that can talk to OpenAI can talk to it — we only change the
# > base URL and the key.
#
# So the same three lines of LangChain code reach either a 2B model on your
# laptop or a 35B model on a GPU cluster. You will see that in notebook 02.
#
# Your key for today lives in `.env`. **Keys are secrets** — don't paste them
# into notebooks, Slack, or git. That is exactly why we keep them in `.env`,
# and why `.env` is listed in `.gitignore`.

# %%
import os
from dotenv import load_dotenv

# Load ../.env into the process environment.
loaded = load_dotenv("../.env")
print("✅ .env loaded" if loaded else "❌ no ../.env found — run: cp ../.env.example ../.env")

BASE_URL = os.getenv("LITELLM_BASE_URL", "")
API_KEY = os.getenv("LITELLM_API_KEY", "")

print("\nLITELLM_BASE_URL :", BASE_URL or "(unset)")
print("LITELLM_API_KEY  :", (API_KEY[:7] + "…" + API_KEY[-4:]) if API_KEY else "(unset)")

# %%
# Which models is *this key* allowed to use? Never assume -- ask the server.
# Keys on a shared proxy are usually scoped to a subset of the models.
def proxy_models(base_url: str, api_key: str) -> list[str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return [m["id"] for m in json.load(resp).get("data", [])]


try:
    allowed = proxy_models(BASE_URL, API_KEY)
    print(f"✅ Proxy reachable. Your key may use {len(allowed)} model(s):\n")
    for name in allowed:
        print("   ", name)
    print("\nPut one of these in .env as LITELLM_CHAT_MODEL")
    print("(or leave it empty and we pick the first one automatically).")
except Exception as exc:
    allowed = []
    print(f"❌ Proxy unreachable or key rejected: {type(exc).__name__}: {exc}")
    print("   No problem — you can do the whole workshop on Ollama.")

# %% [markdown]
# ## 4 · The shared helper, and the actual smoke test
#
# `common/workshop_setup.py` in the repo root wraps everything above into two
# functions we will use for the next nine notebooks:
#
# ```python
# get_chat_model(backend="ollama" | "litellm")   # -> a LangChain chat model
# get_embeddings("nomic-embed-text:latest")      # -> a LangChain embedder
# ```
#
# Open that file at some point today — it is ~150 commented lines and it is the
# only "framework" we hide from you.

# %%
# Make the repo root importable, whether you're running the notebook from this
# folder or the script from somewhere else. Every notebook repeats these 3 lines.
import pathlib
import sys

ROOT = next(
    p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir()
)
sys.path.insert(0, str(ROOT))

from common.workshop_setup import describe_model, get_chat_model, get_embeddings

print("repo root:", ROOT)

# %%
# Smoke test 1: the local model. First call is slow -- Ollama loads weights.
try:
    llm = get_chat_model(backend="ollama", max_tokens=64)
    print("calling", describe_model(llm), "...")
    reply = llm.invoke("Reply with exactly one short sentence: why are you useful?")
    print("✅ Ollama replied:", reply.content.strip()[:300])
except Exception as exc:
    print(f"❌ Ollama call failed: {type(exc).__name__}: {exc}")

# %%
# Smoke test 2: the proxy.
try:
    llm = get_chat_model(backend="litellm", max_tokens=256)
    print("calling", describe_model(llm), "...")
    reply = llm.invoke("Reply with exactly one short sentence: why are you useful?")
    print("✅ Proxy replied:", (reply.content or "").strip()[:300])
except Exception as exc:
    print(f"❌ Proxy call failed: {type(exc).__name__}: {exc}")

# %%
# Smoke test 3: embeddings. A vector of numbers instead of text -- notebook 03.
try:
    embedder = get_embeddings("nomic-embed-text:latest")
    vector = embedder.embed_query("Multi-agent systems are fun.")
    print(f"✅ Got an embedding: {len(vector)} dimensions")
    print("   first 5 values:", [round(v, 4) for v in vector[:5]])
except Exception as exc:
    print(f"❌ Embedding call failed: {type(exc).__name__}: {exc}")

# %% [markdown]
# ## ✅ Checklist before we move on
#
# - [ ] `imprs_workshop` kernel selected, all packages import
# - [ ] at least one of: Ollama replied / proxy replied
# - [ ] an embedding came back with a few hundred dimensions
# - [ ] you know which file holds your model choice (`.env`)
#
# ### Pick your default backend
#
# Open `.env` and set `WORKSHOP_BACKEND` to `ollama` or `litellm`. Every later
# notebook honours it, and you can always override per call with
# `get_chat_model(backend="…")`.
#
# **Rule of thumb for today:** use `ollama` while you are iterating (fast, free,
# no rate limit) and `litellm` when you want the answer to actually be good.
#
# → Next: **02 · Your first LLM call with LangChain**
