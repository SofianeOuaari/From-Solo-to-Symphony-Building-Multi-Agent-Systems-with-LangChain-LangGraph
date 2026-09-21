# 01 · Setting up the environment

**Goal:** every check in the notebook prints ✅ before we touch an agent.

### What it covers
- Creating the environment with **conda** *or* **uv** (both spelled out).
- Registering the `imprs_workshop` Jupyter kernel.
- Verifying every package the workshop needs, one by one.
- Talking to **Ollama** (local models) and listing what you have pulled.
- Talking to the **LiteLLM proxy** and asking it which models *your key* may use.
- Loading `.env`, then three smoke tests: local chat, proxy chat, embeddings.

### Why it's a whole notebook
Half of any workshop's lost time is environment trouble discovered at minute 40.
This front-loads it. It also introduces the two functions used for the rest of
the day: `get_chat_model()` and `get_embeddings()`.

### Before you start
Pull the models — roughly 5 GB, so ideally the day before:

```bash
ollama pull qwen3.5:2b
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text:latest
ollama pull qwen3-embedding:0.6b
ollama pull all-minilm:33m
```

### Exercises
Set `WORKSHOP_BACKEND` in `.env` to your preferred backend, and re-run the
smoke tests. Use `ollama` while iterating, `litellm` when the answer must be
good.

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
