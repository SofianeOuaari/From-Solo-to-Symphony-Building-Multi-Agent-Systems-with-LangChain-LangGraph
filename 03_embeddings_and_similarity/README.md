# 03 · Embeddings and cosine similarity

**The one idea:** sentences with similar meanings get vectors pointing in
similar directions. That single property makes retrieval — and therefore
knowledge bases for agents — possible.

### What it covers
1. One embedding, looked at: 768 numbers, and why no single number means anything.
2. **Cosine similarity derived from the dot product** and implemented in numpy,
   with sanity checks on vectors you can verify by hand.
3. Why normalised vectors turn similarity search into a matrix multiplication.
4. **Pairs where we know the answer** — paraphrases score ~0.95, unrelated
   sentences ~0.40, and *"won the final"* vs *"lost the final"* scores ~0.90.
5. The full similarity matrix, drawn, with the topic blocks marked.
6. Nearest-neighbour search in five lines — retrieval, before we call it that.
7. **Three embedding models compared:** `nomic-embed-text`, `qwen3-embedding:0.6b`
   and `all-minilm:33m` — dimensions, speed, ranking, and multilingual ability.
8. Task prefixes (`search_query:` / `search_document:`) — the gotcha that costs
   you retrieval quality silently.

### The two lessons to insist on
- **Embeddings do not understand negation.** A retriever will hand your agent a
  document stating the exact opposite of what was asked.
- **Cosine values are not comparable across models.** Only the *ranking* is.
  Never pick an embedding model from a leaderboard; build the comparison table
  from your own domain's sentences.

### Exercises
- Add three pairs from your own field. Does the ranking hold?
- Find a pair that *should* be far apart but scores high.
- Redo it with Euclidean distance. Then with normalised vectors — why do they
  now agree?

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
