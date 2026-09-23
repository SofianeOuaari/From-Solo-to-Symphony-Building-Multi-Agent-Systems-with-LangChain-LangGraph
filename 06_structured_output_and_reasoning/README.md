# 06 · Structured output and reasoning

**Two capabilities that stand between a chatbot and an agent.**
Prose is for humans; an agent is a program and needs `route == "billing"`.
And some questions need working-out before the answer.

### What it covers
1. **"Just ask for JSON"** and why 85% reliability is the worst possible number.
2. **Pydantic + `.with_structured_output()`** — a typed object, `Literal`
   constraints, real `bool`s and `list[str]`s, and field descriptions that *are*
   the prompt.
3. **A router in twelve lines** — text in, one-of-N out. This is the supervisor
   of notebook 09.
4. **When validation fails, that is the system working.** Small models answer
   `confidence: 95` meaning percent; Pydantic rejects it loudly instead of
   silently corrupting your data. Three fixes, in order of preference —
   including a `field_validator` that forgives the percent habit.
5. **Nested schemas** — a `Plan` of typed `Step`s. The impressive-sounding
   "planner agent" is a Pydantic model and one `.invoke()`.
6. **Reasoning models** — `deepseek-r1:1.5b` locally, or `thinking=True` on the
   proxy. The hidden chain of thought, printed.
7. **A mini-benchmark**: five tasks × thinking on/off, with accuracy and
   latency. Thinking helps on multi-step problems and buys nothing on lookups.
8. **The tension:** structured output constrains the model to one JSON object;
   thinking wants to ramble. Asked together, they often break.
9. **The pattern that always works:** reason in prose, then structure in a
   second cheap call. Used by essentially every production agent framework.

> ⏱ The benchmark cell runs ten calls against a reasoning model and takes
> several minutes. That slowness *is* the finding. Trim `TASKS` if you're short
> on time.

### Exercises
- Schematise something from your own work and extract it from real text.
- Add two tasks from your field. Does thinking help *there*?
- Find a request where the model correctly reports low confidence. What should
  an agent do with that number?

---

Run it: `jupyter lab notebook.ipynb` — or `python script.py` for the
identical content as a plain script.
