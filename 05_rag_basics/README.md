# 05 · RAG — facts the model was never trained on

**The one idea:** RAG is search plus copy-paste. Find the relevant passages,
paste them into the prompt, ask. Everything else is engineering.

Uses `knowledge_base.md`, the handbook of a **completely fictional** institute —
so no model can have memorised it, which makes retrieval either work or visibly
fail.

### What it covers
1. **The problem, concretely:** ask about `/scratch` retention without the
   document. You usually get a fluent, specific, fabricated answer.
2. **Chunking** with `RecursiveCharacterTextSplitter` — and why `chunk_size`
   is the most underrated knob in RAG, plus what `chunk_overlap` is for.
3. **A vector store in 15 lines** (`TinyVectorStore`) — a matrix and an
   `argsort`. Build it once and vector databases stop being mysterious.
4. **Generation**, and the two prompt lines that do most of the work:
   *"use ONLY the context"* and *"if it's not there, say so"*.
5. The same thing with `InMemoryVectorStore`, `Document` metadata, a retriever,
   and a full streaming LCEL chain.
6. **Breaking it on purpose:**
   - `k` too small for a question spanning two sections
   - short keyword queries reshuffling the ranking unpredictably
   - the retriever *always* returning something, with confident-looking scores,
     for a question the corpus cannot answer
7. The mitigation table (thresholds, hybrid search, rerankers, multi-query) —
   and the observation that several mitigations are *"have an LLM decide
   something first"*, which is the road to agents.

### Exercises
- `chunk_size=150`, then `2000`. Different questions break each time — why those?
- Add a paragraph to the handbook and ask about it.
- Weaken the system prompt until the generator hallucinates from good context.
- Tune the score threshold until it starts refusing questions it should answer.

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
