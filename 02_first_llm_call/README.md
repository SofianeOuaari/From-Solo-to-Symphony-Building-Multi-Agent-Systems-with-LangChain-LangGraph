# 02 · Your first LLM call with LangChain

**The one idea:** a language model is a *function* — text in, text out, plus
randomness. Everything later is scaffolding around it.

### What it covers
1. `llm.invoke("...")`, and what an `AIMessage` actually contains.
2. **Messages and roles** — `SystemMessage` / `HumanMessage` / `AIMessage`. The
   same model becomes three different tools with three different system
   messages. This is where agent *roles* come from.
3. **Memory demystified:** there is none. "Memory" is you resending the earlier
   messages. Shown by running the same question with and without history.
4. `temperature` — the randomness dial, and why agents want it low.
5. **Streaming** — same model, better perceived latency, one line.
6. `ChatPromptTemplate` — reusable, parameterised prompts.
7. **LCEL**: `prompt | model | parser`, and `.batch()` / `.stream()` for free.
8. **Swapping backends** — the identical chain against a 2B local model and a
   35B model on a cluster.
9. **Thinking tokens**, and why a blank reply means "raise `max_tokens`".

### Exercises
- Make the model answer only in bullet points — then make it disobey.
- Build a translation chain and `.batch()` five sentences through it.
- Set `temperature=2.0`. Why is that disastrous for an agent choosing a tool?

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
