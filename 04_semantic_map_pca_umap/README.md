# 04 · Mapping meaning: PCA and UMAP

**The one idea:** we measured semantic structure in notebook 03; now we look at
it — and learn how easily a pretty projection misleads.

### What it covers
1. **Generating a labelled corpus with an LLM** — ~12 short documents for each of
   6 categories (healthcare, sport, education, finance, climate science,
   cooking), cached to `data/generated_corpus.json`.
   Includes a deliberately tolerant parser, because small models mis-format JSON
   constantly. (Notebook 06 shows the proper fix.)
2. Embedding the corpus, and the hygiene of centring and normalising.
3. **PCA** — linear, deterministic, and honest: it *tells you* how much it hid.
   Plus the cumulative-variance curve: ~35 of 768 dimensions carry 90% of the
   variance.
4. **UMAP** — non-linear, prettier, and full of traps: inter-cluster distances
   and cluster sizes are not trustworthy, it is stochastic, and it will show
   you clusters in pure noise.
5. An `n_neighbors` sweep, so participants *feel* the hyperparameter.
6. **Measuring instead of squinting:** within- vs between-category similarity,
   and top-5 neighbour purity — which is exactly the question a retriever asks.
7. The least-pure documents, read by hand. Always the most informative cell.
8. Category centroids, and a **zero-LLM router** built from them.

### Why it matters for agents
That router is real: embed the question, compare it to category prototypes, send
it to the nearest specialist. No LLM call, no latency, no cost — and a low
winning score is your signal to escalate. Notebook 09 does the LLM version.

### Exercises
- Add an overlapping category (`"sports medicine"`). Do the clusters merge?
- Switch to `all-minilm:33m`. Is the 30×-smaller model actually worse *here*?
- **Shuffle the labels and redraw the UMAP plot.** The clusters are still there
  and now mean nothing. Run this control before believing any embedding plot.

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
