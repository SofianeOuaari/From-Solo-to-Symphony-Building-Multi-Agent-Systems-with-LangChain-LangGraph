# 10 · Capstone — the Research Desk

Everything from the previous nine notebooks, in one system.

```
question → PLANNER → RESEARCHER → ANALYST → WRITER ⇄ CRITIC → HUMAN → delivered
                                                     (max 2 revisions)
```

| Component | From |
|-----------|------|
| model access, both backends | 02 |
| embeddings | 03 |
| chunking, vector store, retrieval | 05 |
| `Plan` and `Review` as Pydantic schemas | 06 |
| tools and the agent loop | 07 |
| `StateGraph`, reducers, cycles, checkpointer, `interrupt` | 08 |
| roles, traces, step budgets | 09 |

### What's worth pointing out while it runs
- **Two models on purpose** — a careful one for judgement, a fast one for
  writing. This is where production cost savings live.
- **A retrieval audit log**, so you can show *what was actually retrieved* with
  scores. Provenance is not a nice-to-have: without it nobody can check the
  system's claims.
- **The critic is instructed to reject a draft that answers when the evidence
  says `NOT IN HANDBOOK`.** Each agent hop is a chance to launder a guess into
  a fact; the critic is what stops it.
- **The human is not a rubber stamp.** Run 2 sends feedback back into the graph
  and the writer runs again.

### The three runs
1. A question needing research *and* arithmetic → approved.
2. A question where the human asks for a rewrite → revision loop.
3. A question the handbook cannot answer → an honest non-answer. **The single
   most valuable behaviour in the whole system.**

Then a crude cost/trace accounting, and a note on real tracing (LangSmith or any
OpenTelemetry tracer), which stops being a luxury within about a day.

### What to take away
1. There is no "agent" abstraction — it's a loop around a model that can call
   tools, and a multi-agent system is a graph of those loops over shared state.
2. Roles are prompts, and prompts are specifications that grow from observed
   failures.
3. Structure at the boundaries — Pydantic between agents, not prose.
4. Every cycle needs a budget, every system needs a trace, every consequential
   action needs a human.
5. **The hard part is not the code.** It is deciding what the roles are, who
   needs to see what, and where the system may be wrong. That is research
   design — your job, not the framework's.

### Exercises — pick one and actually build it
1. A **second knowledge source**, so the researcher must choose *which*.
2. A **fact-checker agent** that verifies each claim independently.
3. **Parallel researchers** — fan the plan's steps out, merge with a reducer.
4. **Your own domain.** Swap the handbook for something from your work and
   adjust only the prompts. What broke? That list is the most interesting
   output of the day.

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
