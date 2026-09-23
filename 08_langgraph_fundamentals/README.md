# 08 · LangGraph fundamentals

`create_agent` is fine for one agent with a flat tool list. It is not fine when
you want a critic that sends work back, a router with real branches, a human
approving a step, or several agents handing off. For that you describe the
control flow explicitly.

> **State** — one shared dictionary flowing through the system.
> **Nodes** — functions that read state and return an *update* to it.
> **Edges** — what runs next. Conditional edges let the *data* decide.

### What it covers
1. **A graph with no LLM in it**, so you can predict it completely. Nodes return
   updates; they never mutate state. Then `draw_ascii()` — do this constantly.
2. **Conditional edges** — a function that returns the *name* of the next node.
3. **Reducers** — moving "append, don't overwrite" into the state definition,
   and `add_messages` / `MessagesState`, which nearly every agent graph uses.
4. **A real graph:** triage → route → one of three specialists, each the same
   model with a different system prompt.
5. `.stream()` as your debugger — the state update after every node.
6. **Memory:** a checkpointer plus a `thread_id`. Two lines, and you have
   multi-turn conversations, separate per user, fully inspectable and rewindable.
7. **Cycles** — the writer ⇄ critic loop, which a chain fundamentally cannot do.
   With a hard revision cap, because a cycle without an exit condition is a
   hang. This loop is the heart of notebook 10.
8. **Human-in-the-loop** — `interrupt()` pauses mid-node and hands control back;
   `Command(resume=...)` continues from exactly there.

### Exercises
- Add a fourth category to the support graph. Note that you must change *three*
  places — that is the cost of a hard-coded router, and why notebook 09 differs.
- Make the critic stricter and watch revisions climb. Remove the cap, observe,
  put it back.
- Add a summarisation node that compresses history past 10 messages.
- Let the human return *edits* rather than approve/reject, looping to the writer.

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
