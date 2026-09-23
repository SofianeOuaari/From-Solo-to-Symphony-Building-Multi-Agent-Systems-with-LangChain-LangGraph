# Building Multi-Agent LLM Systems

### A hands-on workshop with LangChain & LangGraph · IMPRS

This hands-on workshop explores how to design and build **multi-agent
collaboration systems**, where multiple LLM-powered agents work together to
tackle tasks. We cover the core concepts behind agentic systems — roles, tools,
memory, knowledge bases — and the orchestration patterns that govern how agents
communicate.

The workshop is a **crescendo**. Notebook 01 checks your Python install;
notebook 10 runs a six-agent research desk with a critic loop and a human
approval gate. Every notebook uses only what the previous ones built, and every
abstraction is written out by hand *before* the library version is shown.

---

## The ten demos

| # | Folder | What you build | Key idea |
|---|--------|----------------|----------|
| 01 | [01_environment_setup/](01_environment_setup/) | a working environment | conda **or** uv; Ollama vs. a LiteLLM proxy |
| 02 | [02_first_llm_call/](02_first_llm_call/) | your first calls and chains | a model is a function; messages, temperature, streaming, LCEL |
| 03 | [03_embeddings_and_similarity/](03_embeddings_and_similarity/) | cosine similarity from scratch | text → vectors; **three** embedding models compared |
| 04 | [04_semantic_map_pca_umap/](04_semantic_map_pca_umap/) | a semantic map of generated documents | LLM-synthesised corpora, PCA vs. UMAP, and how to not fool yourself |
| 05 | [05_rag_basics/](05_rag_basics/) | RAG, including a 15-line vector store | chunking, retrieval, grounding — and the three ways it fails |
| 06 | [06_structured_output_and_reasoning/](06_structured_output_and_reasoning/) | typed outputs and a reasoning benchmark | Pydantic schemas = decisions a program can act on |
| 07 | [07_tools_and_first_agent/](07_tools_and_first_agent/) | the agent loop, written by hand | tools; *the model requests, you execute* |
| 08 | [08_langgraph_fundamentals/](08_langgraph_fundamentals/) | state machines with cycles and memory | `StateGraph`, reducers, checkpointers, `interrupt()` |
| 09 | [09_multi_agent_collaboration/](09_multi_agent_collaboration/) | pipeline, supervisor and network teams | orchestration patterns, and when each is wrong |
| 10 | [10_capstone_research_desk/](10_capstone_research_desk/) | the full system | plan → research → analyse → write → critique → human |

Every folder contains:

* `notebook.ipynb` — for the workshop
* `script.py` — **identical content**, runnable with `python script.py`
* `README.md` — a one-page summary and the exercises

> `script.py` is the source of truth. The two are kept in sync with
> [jupytext](https://jupytext.readthedocs.io/): `jupytext --to notebook script.py`

---

## Setup (do this before the workshop)

### 1 · Python environment — pick **one**

**conda**

```bash
conda create -y -n imprs_workshop python=3.11
conda activate imprs_workshop
pip install -r requirements.txt
python -m ipykernel install --user --name imprs_workshop --display-name "imprs_workshop"
```

**uv** (same result, roughly 10× faster)

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name imprs_workshop --display-name "imprs_workshop"
```

Then `jupyter lab`, and **select the `imprs_workshop` kernel** in each notebook.
Every notebook also carries a `%pip install -r ../requirements.txt` cell, so
people who arrive late or jump ahead are not stuck.

### 2 · Configuration

```bash
cp .env.example .env     # then edit it
```

### 3 · Models

You need **either** Ollama **or** the LiteLLM proxy. Both is better — notebook
02 compares them side by side.

**Ollama** (local, free, private, offline):

```bash
curl -fsSL https://ollama.com/install.sh | sh     # Linux; macOS: ollama.com

ollama pull qwen3.5:2b                # chat model, notebooks 02+
ollama pull deepseek-r1:1.5b          # reasoning model, notebook 06
ollama pull nomic-embed-text:latest   # embeddings, notebooks 03-05, 07-10
ollama pull qwen3-embedding:0.6b      # embeddings comparison, notebook 03
ollama pull all-minilm:33m            # embeddings comparison, notebook 03
```

≈ 5 GB in total. **Start this download the day before.** Smaller substitutes are
fine (`qwen3.5:0.8b`) change the name in `.env` and every
notebook follows.

**LiteLLM proxy** (a shared server, bigger models, needs a key): put the base
URL and key in `.env`. Notebook 01 prints exactly which models *your* key may
use — don't assume, the proxy will tell you.

Switch between them with one line in `.env`:

```ini
WORKSHOP_BACKEND=ollama      # or: litellm
```

or per call, anywhere: `get_chat_model(backend="litellm")`.

**Embeddings are always local.** They are tiny, fast, and comparing three of
them is the point of notebook 03.

---

## How the code is organised

```
IMPRS_LLM_Workshop/
├── common/workshop_setup.py    ← the ONLY shared helper. ~200 commented lines.
├── requirements.txt
├── .env                        ← your model choice and your key (gitignored)
├── data/                       ← generated corpora and figures (gitignored)
├── 01_environment_setup/ … 10_capstone_research_desk/
└── 05_rag_basics/knowledge_base.md   ← the fictional handbook used from 05 onward
```

`common/workshop_setup.py` gives you three functions. That is the entire
framework we hide:

```python
get_chat_model(backend=None, model=None, temperature=0.2,
               max_tokens=1024, thinking=False)
get_reasoning_model()                     # same, with thinking switched on
get_embeddings("nomic-embed-text:latest")
```

Notebooks start with three lines that make the repo importable from anywhere:

```python
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
            if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))
from common.workshop_setup import get_chat_model
```

### The knowledge base

`05_rag_basics/knowledge_base.md` is the handbook of a **completely fictional**
institute. That is deliberate: no model has memorised it, so retrieval either
works or visibly doesn't. Notebooks 05, 07, 09 and 10 all use it.

---

## Known behaviour on a small local model

Everything here was run end to end against `qwen3.5:2b` on a laptop. Some cells
*misbehave in instructive ways*, and the notebooks say so where it happens:

| Where | What you will see | Why it is in the workshop |
|-------|-------------------|---------------------------|
| 03 | "won the final" ≈ "lost the final" | embeddings do not encode negation |
| 04 | PCA shows 2 clusters, UMAP shows 6 | projections are not evidence |
| 05 | confident scores for unanswerable questions | retrievers always return *something* |
| 06 | `confidence: 95` rejected by Pydantic | validation catching a real error |
| 06 | wrong answers in the reasoning benchmark | a 2B model needs tools, not thinking |
| 07 | the vague-docstring tool is never called | the docstring *is* the prompt |
| 07 | tool returns 21495, model says "€86" | tool use ≠ correct reporting |
| 08 | the stats question routes to `technical` | a router caps everything downstream |
| 09 | the network pattern hits its hop limit | why supervisors are the default |
| 10 | the critic approves a self-contradiction | a critic raises the floor, not a guarantee |

Switching `WORKSHOP_BACKEND=litellm` fixes most of them — which is itself the
most memorable lesson of the day about model size.

## Regenerating the notebooks

`script.py` is the source of truth; `notebook.ipynb` is generated from it.

```bash
./sync_notebooks.sh          # script.py  -> notebook.ipynb
./sync_notebooks.sh --back   # notebook.ipynb -> script.py (after editing in Jupyter)
```
