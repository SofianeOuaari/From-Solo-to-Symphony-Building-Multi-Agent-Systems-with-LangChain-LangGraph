"""
Shared plumbing for the IMPRS multi-agent workshop.
===================================================

Every notebook starts by importing from here. The point of this file is that
**you only configure your model once**, in `.env`, and then all ten demos run
on whichever backend you chose:

    ollama   -> a model running on your own laptop, private and free
    litellm  -> a shared OpenAI-compatible proxy, bigger models, needs a key

The functions are deliberately short. Read them once; there is no magic.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Where are we?
# --------------------------------------------------------------------------

def repo_root() -> Path:
    """Walk upwards from this file until we find the workshop root."""
    return Path(__file__).resolve().parent.parent


def load_env(verbose: bool = False) -> None:
    """Read `.env` into os.environ. Safe to call many times."""
    env_file = repo_root() / ".env"
    if not env_file.exists():
        raise FileNotFoundError(
            f"No .env found at {env_file}.\n"
            "Run:  cp .env.example .env   and fill it in (see notebook 01)."
        )
    load_dotenv(env_file, override=False)
    if verbose:
        print(f"Loaded config from {env_file}")


load_env()

DATA_DIR = repo_root() / "data"
DATA_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Chat models
# --------------------------------------------------------------------------

def get_chat_model(
    backend: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    thinking: bool = False,
    **kwargs,
):
    """Return a LangChain chat model.

    Parameters
    ----------
    backend : "ollama" | "litellm" | None
        None  -> whatever WORKSHOP_BACKEND says in `.env`.
    model : str | None
        None  -> the default model for that backend, from `.env`.
    temperature : float
        0.0 = as deterministic as the model gets, 1.0 = creative.
    max_tokens : int
        Upper bound on the *answer* length. Keep it generous for reasoning
        models: their internal "thinking" is charged against this budget, so a
        small value can leave you with an empty answer. (We demo that in 06.)
    thinking : bool
        Most models we use today (Qwen3.5, Qwen3.6, DeepSeek-R1) can write a
        long private chain of thought before answering. It often helps accuracy,
        but it is slow and it eats your `max_tokens` budget -- so we default it
        to **off** and switch it on deliberately in notebook 06.

    The returned object always speaks the same LangChain interface:
    `.invoke()`, `.stream()`, `.bind_tools()`, `.with_structured_output()`.
    That interchangeability is the whole reason we bother with LangChain.
    """
    backend = (backend or os.getenv("WORKSHOP_BACKEND", "ollama")).lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
            num_predict=max_tokens,   # Ollama's name for max_tokens
            reasoning=thinking,       # Ollama's name for "think before answering"
            **kwargs,
        )

    if backend == "litellm":
        from langchain_openai import ChatOpenAI

        # A LiteLLM proxy is just an OpenAI-compatible HTTP endpoint, so the
        # OpenAI client talks to it happily -- we only swap the base URL.
        # `extra_body` is how the OpenAI client smuggles vendor-specific options
        # through to the server -- here, Qwen's thinking toggle.
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        extra_body.update(kwargs.pop("extra_body", {}))

        return ChatOpenAI(
            model=model or _default_litellm_model(),
            base_url=os.environ["LITELLM_BASE_URL"],
            api_key=os.environ["LITELLM_API_KEY"],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            **kwargs,
        )

    raise ValueError(f"Unknown backend {backend!r}. Use 'ollama' or 'litellm'.")


def get_reasoning_model(backend: str | None = None, **kwargs):
    """A model with `thinking` switched ON (notebook 06).

    On Ollama this also swaps in a dedicated reasoning model (DeepSeek-R1);
    on the proxy the same model simply starts thinking out loud.
    """
    backend = (backend or os.getenv("WORKSHOP_BACKEND", "ollama")).lower()
    model = os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:1.5b") if backend == "ollama" else None
    kwargs.setdefault("max_tokens", 2048)   # thinking needs room
    kwargs["thinking"] = True
    return get_chat_model(backend=backend, model=model, **kwargs)


def litellm_models() -> list[str]:
    """Ask the proxy which models *this key* is allowed to use."""
    import urllib.request
    import json

    url = os.environ["LITELLM_BASE_URL"].rstrip("/") + "/models"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    return [m["id"] for m in payload.get("data", [])]


def _default_litellm_model() -> str:
    configured = os.getenv("LITELLM_CHAT_MODEL", "").strip()
    if configured:
        return configured
    models = litellm_models()
    if not models:
        raise RuntimeError("The proxy offered no models for this key.")
    return models[0]


# --------------------------------------------------------------------------
# Embedding models
# --------------------------------------------------------------------------

def get_embeddings(model: str = "nomic-embed-text:latest", **kwargs):
    """Return an Ollama embedding model.

    Embeddings stay local for the whole workshop on purpose: the models are
    tiny (a few hundred MB), they run in milliseconds, and it lets us compare
    three of them side by side without burning shared quota.
    """
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        **kwargs,
    )


def embed_model_names() -> list[str]:
    raw = os.getenv(
        "EMBED_MODELS",
        "nomic-embed-text:latest,qwen3-embedding:0.6b,all-minilm:33m",
    )
    return [m.strip() for m in raw.split(",") if m.strip()]


# --------------------------------------------------------------------------
# Small conveniences used across the notebooks
# --------------------------------------------------------------------------

def describe_model(llm) -> str:
    """One-line human description of whatever model object you hand it."""
    name = getattr(llm, "model", None) or getattr(llm, "model_name", "?")
    return f"{type(llm).__name__}(model={name!r})"


def show(response, label: str = "answer") -> str:
    """Print a chat response nicely and hand back its text.

    Reasoning models sometimes return an empty `content` because they spent
    their whole token budget thinking. We surface that instead of printing
    a confusing blank line.
    """
    text = response.content if hasattr(response, "content") else str(response)
    thinking = ""
    if hasattr(response, "additional_kwargs"):
        extra = response.additional_kwargs
        # OpenAI-compatible servers call it reasoning_content; Ollama calls it
        # reasoning_content too, via langchain-ollama. Check both spellings.
        thinking = extra.get("reasoning_content") or extra.get("reasoning") or ""

    if thinking:
        print(f"--- hidden thinking ({len(thinking)} chars) ---")
        print(thinking[:500] + ("..." if len(thinking) > 500 else ""))
    print(f"--- {label} ---")
    print(text if text else "(empty! raise max_tokens -- see notebook 06)")

    usage = getattr(response, "usage_metadata", None)
    if usage:
        print(
            f"\n[tokens: {usage.get('input_tokens')} in / "
            f"{usage.get('output_tokens')} out]"
        )
    return text


def _in_notebook() -> bool:
    """True only inside a Jupyter kernel, not in a plain script or IPython shell."""
    try:
        from IPython import get_ipython

        return type(get_ipython()).__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def draw(graph, png: bool = True) -> None:
    """Show a compiled LangGraph graph.

    Tries, in order: an inline PNG (notebooks, needs network), ASCII art
    (needs `grandalf`), and finally the raw Mermaid source -- which you can
    always paste into https://mermaid.live.
    """
    representation = graph.get_graph()

    if png and _in_notebook():
        try:
            from IPython.display import Image, display

            display(Image(representation.draw_mermaid_png()))
            return
        except Exception:
            pass   # no network for the mermaid renderer -- fall through

    try:
        print(representation.draw_ascii())
        return
    except Exception:
        pass       # grandalf not installed

    print(representation.draw_mermaid())


class timer:
    """`with timer("embedding"):` -> prints how long the block took."""

    def __init__(self, label: str = "block"):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        print(f"[{self.label}: {self.seconds:.2f}s]")
        return False
