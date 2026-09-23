# 09 · Multi-agent collaboration

The actual subject of the workshop — and it opens with the argument *against*
it.

### Start here: why bother?
One agent with fifteen tools is simpler, cheaper and easier to debug than five
agents with three each. Multi-agent is worth it for specific, nameable reasons:
tool-selection accuracy, **context isolation**, role clarity, different models
per role, parallelism, separable evaluation.

The costs are real too: latency and tokens per handoff, compounding failures
(four agents at 90% each is 66% end-to-end), harder debugging, and agents that
loop or politely hand work back and forth forever.

> **Rule of thumb: start with one agent. Split it when you can name which
> reason applies.** Make participants justify every split.

### The four patterns
1. **Pipeline** — researcher → analyst → writer. *You* decided the flow. Shows
   context isolation beautifully (the writer never sees a retrieved chunk), and
   its one weakness: it runs all three agents even for "what's 2+2".
2. **Supervisor** — a router agent picks who works next, repeatedly, against a
   shared scratchpad, until it says FINISH. The workhorse: routing logic lives
   in one readable place, and adding a specialist is a node plus a description.
   Run on three questions — one needing research *and* arithmetic, one needing
   neither, and one the knowledge base cannot answer.
3. **Network** — agents hand off directly with `Command(goto=...)`. No
   supervisor call per hop, much harder to reason about.
4. **Hierarchical** — supervisors of supervisors. Discussed, not built: it is
   the same code twice. Only worth it past ~7 workers.

Plus a comparison table: who decides the flow, what each is best for, and the
main risk of each.

### The three rules that keep these systems working
1. **Every cycle needs a hard budget.** Not defensive programming — a
   requirement. Agents loop.
2. **Print the trace.** You cannot debug a multi-agent system from its final
   answer.
3. **Narrow roles, structured outputs.** "Report only what the handbook says" is
   a testable contract. "Be helpful" is not.

### Exercises
- Add a **Critic** worker to the supervisor team (you built it in notebook 08).
- Put the supervisor on `litellm` and the workers on `ollama`. Where does
  quality change most?
- Delete "Use LAST" from the writer's description. Does the supervisor now call
  it first?
- Ask something needing two separate lookups. Does the librarian get called twice?
- Set the step budget to 2. Is the failure obvious to the user? It should be.

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
