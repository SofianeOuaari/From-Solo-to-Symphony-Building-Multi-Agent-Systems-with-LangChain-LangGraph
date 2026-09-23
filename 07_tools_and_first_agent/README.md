# 07 · Tools, and your first agent

> **A chain is a fixed sequence of steps that you wrote.
> An agent is a loop where the *model* decides the next step.**

That is the whole difference. Memory, roles and multi-agent orchestration are
all built on this loop.

### What it covers
1. **A tool is a function with a good docstring.** The model sees the name, the
   docstring and the type hints — nothing else. So the docstring *is the
   prompt*, and a vague one is the #1 reason "my agent ignores my tool".
2. **The model requests; it does not execute.** `bind_tools` returns an
   `AIMessage` with `tool_calls` and empty content. Nothing has run. *You* run
   it — which is also your safety boundary. **This is the most commonly
   misunderstood point in the whole workshop.**
3. **The agent loop, by hand** — 20 lines, no framework:
   ask the model → tool calls? → run them, append `ToolMessage`s, repeat.
   With a `max_steps` guard, which is not optional.
4. The same thing via `create_agent` — and the note that this is *already* a
   LangGraph graph: a two-node model ⇄ tools cycle.
5. **Retrieval as a tool.** In notebook 05 the chain always searched. Now the
   agent decides whether to search, what to search for, and whether once was
   enough. Watch it search, then call the calculator on what it found — two
   tools, chosen and ordered by the model.
6. **The four ways agents fail:** the unused tool (bad docstring), the erroring
   tool (a good agent recovers), the infinite loop, and **too many tools**.

### The punchline
Past roughly 10–15 tools, tool-selection accuracy drops. That is not a bug to
fix — it is *the argument for multi-agent systems*. Five agents with eight tools
each, and a supervisor routing between them.

### Exercises
- Write a tool that does something real in your work; get the agent to use it.
- Shorten `search_handbook`'s docstring to three words. Does it still search?
  Now make it too broad — does it search when it shouldn't?
- Ask something needing three tools in sequence. Where does it break?
- Set `temperature=1.0` and run the same question five times. How consistent is
  tool selection — and what temperature should an agent run at?

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
