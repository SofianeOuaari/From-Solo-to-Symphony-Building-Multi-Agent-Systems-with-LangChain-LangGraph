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
# # 05 · RAG — giving a model facts it was never trained on
#
# A language model knows what was in its training data, and nothing else. It
# does not know your lab's handbook, your unpublished results, or anything that
# happened after its cutoff. Worse: when asked, it will often **invent** a
# confident answer rather than admit ignorance.
#
# **Retrieval-Augmented Generation (RAG)** is the fix, and it is much simpler
# than the acronym suggests:
#
# ```
# question ──► find the relevant passages ──► paste them into the prompt ──► ask
#                    (notebook 03!)                                        (notebook 02!)
# ```
#
# That's it. RAG is search plus copy-paste. Everything else is engineering.
#
# We build it in five steps:
#
# 1. see the model hallucinate, so the problem is concrete
# 2. **chunk** a document into retrievable pieces
# 3. **index** those chunks (by hand with numpy, then with LangChain)
# 4. **retrieve** and **generate**
# 5. break it on purpose, and learn where RAG fails

# %%
# %pip install -q -r ../requirements.txt

# %%
import pathlib
import sys

ROOT = next(
    p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir()
)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.workshop_setup import describe_model, get_chat_model, get_embeddings, timer

llm = get_chat_model(temperature=0.1, max_tokens=500)
embedder = get_embeddings("nomic-embed-text:latest")
print("chat :", describe_model(llm))

# %% [markdown]
# ## 1 · The problem, made concrete
#
# Next to this notebook sits `knowledge_base.md`: the internal handbook of a
# **completely fictional** research institute. No model on earth has read it.
# That's the point — it is an honest test.
#
# Let's ask a question that can only be answered from that document.

# %%
QUESTION = "On the HELIOS cluster, how long are files kept on /scratch before deletion?"

print("WITHOUT the document:\n")
print(llm.invoke(QUESTION).content)

# %% [markdown]
# Read what came back. Depending on the model you will get either an honest
# "I don't know which cluster you mean", or — more often — a fluent, specific,
# **completely fabricated** answer, e.g. "typically 90 days".
#
# That second failure mode is the dangerous one, and it is why RAG exists: not
# to make the model smarter, but to make it *accountable to a source*.

# %% [markdown]
# ## 2 · Chunking
#
# Why not just paste the whole document into the prompt? For a 3-page handbook
# you could. For 3,000 papers you cannot — context windows are finite, long
# contexts are expensive, and models get measurably worse at finding a fact
# buried in the middle of a huge prompt.
#
# So we split the document into **chunks** and retrieve only the relevant ones.
#
# Chunking is the single most underrated knob in RAG:
#
# * **too small** → a chunk arrives without the context that makes it meaningful
#   ("...is 30 days." — 30 days of *what*?)
# * **too large** → the chunk is mostly irrelevant text, which dilutes its
#   embedding and wastes prompt space
#
# `RecursiveCharacterTextSplitter` splits on the *largest* separator that fits:
# paragraphs first, then lines, then sentences, then words. So it breaks at
# natural boundaries whenever it can.

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = (pathlib.Path.cwd() / "knowledge_base.md").read_text()
print(f"document: {len(document)} characters, {len(document.split())} words")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,        # target size in characters
    chunk_overlap=100,     # repeat the last 100 chars in the next chunk...
    separators=["\n## ", "\n\n", "\n", ". ", " "],   # ...prefer section breaks
)
chunks = splitter.split_text(document)

print(f"→ {len(chunks)} chunks")
print(f"  sizes: min={min(map(len, chunks))}, "
      f"median={int(np.median([len(c) for c in chunks]))}, "
      f"max={max(map(len, chunks))}")

# %%
# Always look at your chunks. Bad retrieval is usually bad chunking.
for i in (0, 5, len(chunks) - 1):
    print(f"\n{'─' * 70}\nCHUNK {i}\n{'─' * 70}\n{chunks[i]}")

# %% [markdown]
# > **Why `chunk_overlap`?** A fact that straddles a boundary would otherwise be
# > cut in half and become unretrievable. Overlap means every sentence appears
# > whole in at least one chunk. It costs a little redundancy and buys a lot of
# > robustness.

# %% [markdown]
# ## 3 · Indexing — the whole vector store, in 15 lines
#
# Before reaching for Chroma / FAISS / pgvector, build one yourself. A vector
# store is a matrix and an `argsort`. Understanding that means you will never be
# confused by a vector database again.

# %%
class TinyVectorStore:
    """A vector store with nothing hidden: a matrix, and a dot product."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.texts: list[str] = []
        self.matrix: np.ndarray | None = None

    def add(self, texts: list[str]) -> "TinyVectorStore":
        vectors = np.array(self.embedder.embed_documents(texts))
        # Normalise once, at write time, so search is a plain dot product.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.texts += texts
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        return self

    def search(self, query: str, k: int = 4) -> list[tuple[float, str]]:
        q = np.array(self.embedder.embed_query(query))
        q /= np.linalg.norm(q)
        scores = self.matrix @ q                       # cosine with every chunk
        best = np.argsort(-scores)[:k]                 # top k, highest first
        return [(float(scores[i]), self.texts[i]) for i in best]


with timer(f"indexing {len(chunks)} chunks"):
    store = TinyVectorStore(embedder).add(chunks)

print("index shape:", store.matrix.shape, "→ (chunks, dimensions)")

# %%
# Does it find the right passage?
for score, chunk in store.search(QUESTION, k=3):
    print(f"\n[{score:.3f}] {chunk[:220].strip()}…")

# %% [markdown]
# ## 4 · Generation — the prompt is where RAG succeeds or fails
#
# We have the passages. Now we paste them into a prompt. Two instructions do
# almost all the work:
#
# * **"use ONLY the context"** — stops the model blending in half-remembered
#   training data
# * **"if the context doesn't contain the answer, say so"** — gives it a legal
#   way out, so it doesn't have to invent one
#
# Without that second line, a model under pressure to be helpful will fabricate.

# %%
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the Institute for Adaptive Systems.\n"
            "Use ONLY the context below. Do not use prior knowledge.\n"
            "If the context does not contain the answer, reply exactly:\n"
            "  'The handbook does not cover this.'\n"
            "Quote the relevant numbers verbatim. Be concise.\n\n"
            "--- CONTEXT ---\n{context}\n--- END CONTEXT ---",
        ),
        ("human", "{question}"),
    ]
)


def answer(question: str, k: int = 4, verbose: bool = False) -> str:
    hits = store.search(question, k=k)
    context = "\n\n---\n\n".join(text for _, text in hits)

    if verbose:
        print(f"retrieved {k} chunks, scores: {[round(s, 3) for s, _ in hits]}")
        print(f"context length: {len(context)} characters\n")

    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


print("WITH the document:\n")
print(answer(QUESTION, verbose=True))

# %% [markdown]
# Compare that to section 1. Same model, same question — the only difference is
# that we found the right paragraph and pasted it in first.

# %%
# A handful of questions, including one the handbook deliberately cannot answer.
questions = [
    "Who directs the institute and since when?",
    "How much conference money does a PhD student get per year?",
    "How many GPUs does one HELIOS node have?",
    "What happens if I start a human study without ethics approval?",
    "How long does a PhD defence last, and how is the time split?",
    "What is the institute's policy on remote work?",   # ← not in the handbook
]

for question in questions:
    print(f"\nQ: {question}\nA: {answer(question).strip()}")

# %% [markdown]
# The last one is the one to celebrate. **Knowing when you don't know** is worth
# more than any benchmark score — and in notebook 09 it is what lets a
# supervisor agent decide to ask a *different* specialist instead of shipping a
# guess.

# %% [markdown]
# ## 5 · The same thing, the LangChain way
#
# Now that you know what's underneath, use the library version. It gives you the
# same behaviour plus metadata, filtering, persistence, and a swap-in path to a
# real database (Chroma, FAISS, pgvector, Qdrant…) behind an identical interface.

# %%
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

# Documents carry metadata -- which is how you later show citations, or filter
# by year / author / access level before searching.
documents = [
    Document(page_content=chunk, metadata={"source": "knowledge_base.md", "chunk": i})
    for i, chunk in enumerate(chunks)
]

vector_store = InMemoryVectorStore(embedder)
with timer("indexing with LangChain"):
    vector_store.add_documents(documents)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

hits = retriever.invoke("How do I get a HELIOS account?")
for hit in hits:
    print(f"[chunk {hit.metadata['chunk']}] {hit.page_content[:110].strip()}…")

# %% [markdown]
# ### The full chain, in one expression
#
# A retriever is a Runnable, so it pipes like everything else. `RunnablePassthrough`
# just forwards the original question alongside the retrieved context.

# %%
from langchain_core.runnables import RunnablePassthrough


def format_documents(documents) -> str:
    return "\n\n---\n\n".join(
        f"[chunk {d.metadata['chunk']}]\n{d.page_content}" for d in documents
    )


rag_chain = (
    {"context": retriever | format_documents, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What are the rules for storing personal data?"))

# %%
# It streams, too -- the whole pipeline, end to end.
for piece in rag_chain.stream("What must happen within 72 hours?"):
    print(piece, end="", flush=True)
print()

# %% [markdown]
# ## 6 · Breaking it on purpose
#
# RAG demos always work. RAG systems often don't. Here are the three failures
# you will actually meet.

# %% [markdown]
# ### Failure 1 — `k` is too small
#
# If the answer needs facts from two different sections, retrieving one chunk
# cannot work. No prompt engineering saves you; the information is not there.

# %%
# This one genuinely needs two different sections of the handbook: the
# department list AND the publication policy.
multi_hop = (
    "Who leads the department that runs the Lake Constance field stations, "
    "and who mediates authorship disputes?"
)

for k in [1, 2, 6]:
    print(f"\n── k={k} ──")
    print(answer(multi_hop, k=k).strip()[:300])

# %% [markdown]
# ### Failure 2 — short queries are unstable
#
# Embeddings handle synonyms well, but they need something to work with. Very
# short keyword-style queries carry little signal, and the ranking becomes
# fragile: re-word the query slightly and a different chunk wins.
#
# All five queries below are asking the same thing — *when does /scratch get
# purged?* Watch which chunk each one actually lands on, and how close the
# scores are to each other.

# %%
for query in [
    "scratch",
    "scratch file retention",
    "purge schedule",
    "how long until my temp stuff vanishes",
    "After how many days are files on /scratch automatically deleted?",
]:
    top2 = store.search(query, k=2)
    print(f"\n» {query!r}")
    for score, chunk in top2:
        heading = chunk.strip().splitlines()[0][:58]
        hit = "✅" if "/scratch" in chunk else "  "
        print(f"   {hit} {score:.3f}  {heading}")

# %% [markdown]
# Two things to take away. First, the *full question* at the bottom usually
# wins — longer queries embed better than keywords, which is why "query
# expansion" is a standard RAG trick. Second, look at the **gap** between the
# top-1 and top-2 scores: when it is tiny, the ranking is essentially a
# coin flip, and `k=1` would be reckless.

# %% [markdown]
# ### Failure 3 — the retriever always returns *something*
#
# This is the one that bites in production. Cosine similarity has no concept of
# "nothing matched". Ask about a topic the corpus knows nothing about and you
# still get four chunks back, with respectable-looking scores.

# %%
for score, chunk in store.search("What is the airspeed velocity of a swallow?", k=3):
    print(f"[{score:.3f}] {chunk[:80].strip()}…")

print("\n↑ Confident scores for a question the handbook cannot answer.")
print("  The *prompt* is what saves us here, not the retriever:")
print("\n ", answer("What is the airspeed velocity of a swallow?").strip())

# %% [markdown]
# ### The standard mitigations
#
# | Problem | Fix |
# |---------|-----|
# | irrelevant chunks retrieved | **score threshold** — drop hits below ~0.5 |
# | vocabulary mismatch | **hybrid search** (BM25 keyword + vector), or ask an LLM to rewrite the query first |
# | right chunk retrieved but ranked 8th | **reranker** — retrieve 20, have a cross-encoder re-score, keep 4 |
# | answer needs several sections | raise `k`, or **multi-query**: generate 3 paraphrases and merge results |
# | model ignores the context anyway | stricter prompt, lower temperature, demand quotes |
#
# Notice that several of those fixes are *"have an LLM do something clever
# first"*. Query rewriting is an LLM step. Reranking is a model. Deciding
# whether to retrieve at all is a decision.
#
# **That is exactly the road to agents.** A fixed chain does the same thing every
# time. An agent *chooses*: rewrite the query, search again, search a different
# source, or answer directly. We give it that ability in notebook 07.

# %%
# A 5-line score threshold -- the cheapest reliability win in RAG.
def answer_with_threshold(question: str, k: int = 4, threshold: float = 0.5) -> str:
    hits = [(s, t) for s, t in store.search(question, k=k) if s >= threshold]
    if not hits:
        return f"No chunk scored above {threshold}. Refusing to answer."
    context = "\n\n---\n\n".join(t for _, t in hits)
    return (rag_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )


print(answer_with_threshold("What is the airspeed velocity of a swallow?"))
print()
print(answer_with_threshold("How many ECTS are mandatory in the PhD programme?"))

# %% [markdown]
# ## Your turn — 10 minutes
#
# 1. Set `chunk_size=150` and re-index. Which questions break, and why? Now try
#    `chunk_size=2000`. Different questions break — why those?
# 2. Add a paragraph to `knowledge_base.md` about something the institute does,
#    re-run the indexing cells, and ask about it.
# 3. Write a question that the retriever answers correctly but you can make the
#    *generator* get wrong by weakening the system prompt. (Try deleting the
#    "use ONLY the context" line.)
# 4. Tune `threshold` in the last cell. Where does it start rejecting questions
#    it should have answered? That trade-off never goes away.
#
# → Next: **06 · Structured output and reasoning** — how to make a model return
# something a *program* can use, not just something a human can read.
