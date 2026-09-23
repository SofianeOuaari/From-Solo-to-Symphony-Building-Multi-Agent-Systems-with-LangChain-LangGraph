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
# # 06 · Structured output and reasoning
#
# Two capabilities stand between "a chatbot" and "an agent". This notebook is
# about both.
#
# **Structured output.** A prose answer is for a human. An agent is a *program*:
# it needs `route == "billing"`, not "Well, I'd probably send this to billing,
# though you could argue…". We need the model to fill in a form.
#
# **Reasoning.** Some questions need working-out before the answer. Modern
# models can produce a private chain of thought first. It costs time and tokens;
# we will measure whether it is worth it.
#
# These two are also in *tension* — and knowing when to use which is a real
# design decision you will face in every agent you build.

# %%
# %pip install -q -r ../requirements.txt

# %%
import pathlib
import sys

ROOT = next(
    p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir()
)
sys.path.insert(0, str(ROOT))

import json

import pandas as pd

from common.workshop_setup import (
    describe_model,
    get_chat_model,
    get_reasoning_model,
    show,
    timer,
)

llm = get_chat_model(temperature=0, max_tokens=600)
print("using:", describe_model(llm))

# %% [markdown]
# ## 1 · The naive way, and why it hurts
#
# Everyone's first attempt is "just ask for JSON". It works about 85% of the
# time, which is the worst possible number: often enough to ship, rare enough to
# page you at 3 a.m.

# %%
EMAIL = """
Hi, this is Sandra Kowalski from the Bavarian Forest field station.
Our two HOBO temperature loggers (serials H-4471 and H-4489) stopped writing to
the SD card sometime around the 3rd of March. We've lost roughly six weeks of
canopy data. This is fairly urgent -- the growing-season analysis is due to my
committee in April. Could someone from Environmental Sensing take a look?
"""

naive = llm.invoke(
    f"Extract the sender, urgency (low/medium/high) and a one-line summary "
    f"from this email as JSON:\n\n{EMAIL}"
)
print(naive.content)

# %%
# Now try to actually *use* it:
try:
    parsed = json.loads(naive.content)
    print("✅ parsed:", parsed)
except json.JSONDecodeError as exc:
    print(f"❌ json.loads failed: {exc}")
    print("\nTypical culprits: ```json fences, a 'Here is the JSON:' preamble,")
    print("a trailing comma, or a helpful closing remark after the object.")

# %% [markdown]
# You *can* patch this with `.strip()`, regex, and retries. Please don't.

# %% [markdown]
# ## 2 · The right way: describe a schema, get an object
#
# Define the shape you want with **Pydantic**, hand it to
# `.with_structured_output()`, and get back a typed Python object — validated,
# with defaults, with docstrings that double as instructions to the model.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class SupportTicket(BaseModel):
    """A structured summary of an incoming support email."""

    sender: str = Field(description="full name of the person writing")
    department: Literal[
        "Collective Behaviour", "Computational Cognition",
        "Environmental Sensing", "Methods and Statistics", "unknown",
    ] = Field(description="which department should handle this")
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(description="one sentence, max 20 words")
    equipment: list[str] = Field(
        default_factory=list, description="serial numbers or device names mentioned"
    )
    data_loss: bool = Field(description="true if any data was lost")


extractor = llm.with_structured_output(SupportTicket)

with timer("structured extraction"):
    ticket = extractor.invoke(EMAIL)

print(type(ticket).__name__, "\n")
for field, value in ticket.model_dump().items():
    print(f"  {field:<12} = {value!r}")

# %% [markdown]
# Look at what we got for free:
#
# * a real Python object — `ticket.urgency`, with autocomplete
# * `Literal[...]` **constrains** the department to a valid value; no more
#   "Env. Sensing" vs "environmental_sensing" vs "Environmental Sensing Dept."
# * a typed `list[str]` and a real `bool`, not the string `"true"`
# * the field descriptions are sent to the model as instructions, so the schema
#   *is* the prompt
#
# Under the hood LangChain uses tool-calling (or the provider's JSON mode) to
# force the model's output to fit. That's why it is reliable in a way that
# "please return JSON" never is.

# %%
# It's a normal object -- so normal code can branch on it.
if ticket.urgency == "high" and ticket.data_loss:
    print(f"🚨 escalate to {ticket.department}")
    print(f"   affected hardware: {', '.join(ticket.equipment) or 'none listed'}")

# %% [markdown]
# ## 3 · Structured output *is* how agents make decisions
#
# Keep that `Literal` in mind. An agent's router is exactly this: text in,
# one-of-N-choices out. Here is a router in twelve lines — we will meet it again
# in notebook 09 as the supervisor of a multi-agent system.

# %%
class Routing(BaseModel):
    """Which specialist should handle this request, and how sure are we."""

    destination: Literal["retrieval", "computation", "writing", "smalltalk"]
    reasoning: str = Field(description="one short sentence justifying the choice")
    confidence: float = Field(
        ge=0.0, le=1.0,
        # Spell the range out! A model that only sees `le=1.0` in a schema will
        # still happily answer 95, meaning "95%". Say it in words.
        description="how certain you are, as a decimal between 0.0 and 1.0",
    )


router = llm.with_structured_output(Routing)

REQUESTS = [
    "What does our handbook say about storing personal data?",
    "What's 18.4 million divided by 214 employees?",
    "Draft a polite email declining a reviewing invitation.",
    "Morning! How's it going?",
]

for request in REQUESTS:
    try:
        decision = router.invoke(request)
        print(f"{decision.destination:<12} (conf {decision.confidence:.2f})  ← {request[:48]}")
        print(f"             {decision.reasoning}\n")
    except Exception as exc:
        print(f"❌ {type(exc).__name__} on {request[:40]!r}")
        print(f"   {str(exc)[:160]}\n")

# %% [markdown]
# ### When validation fails, that is the system working
#
# Small models violate constraints. A favourite: `confidence: 95`, because the
# model is thinking in percent. Pydantic rejects it and you get a loud,
# specific error.
#
# **That is the whole point.** Without a schema you would have silently stored
# `95` in a field your code compares against `0.7`, and every request would look
# maximally confident forever. A crash at the boundary beats corrupt data
# downstream.
#
# Three ways to deal with it, in order of preference:
#
# 1. **Describe the constraint in words**, not just in the type
#    (`description="a decimal between 0.0 and 1.0"`). Usually enough.
# 2. **Loosen the type and normalise yourself** — accept any float, then
#    `value / 100 if value > 1 else value`.
# 3. **Retry.** `llm.with_structured_output(Routing).with_retry()` re-asks on a
#    parse failure. Costs a call; fixes most transient formatting slips.

# %%
# Option 2, in practice -- a validator that forgives a common model habit.
from pydantic import field_validator


class ForgivingRouting(BaseModel):
    destination: Literal["retrieval", "computation", "writing", "smalltalk"]
    confidence: float

    @field_validator("confidence")
    @classmethod
    def percent_to_fraction(cls, value: float) -> float:
        return value / 100 if value > 1 else value


forgiving = llm.with_structured_output(ForgivingRouting)
for request in REQUESTS[:2]:
    try:
        decision = forgiving.invoke(request)
        print(f"{decision.destination:<12} conf={decision.confidence:.2f}  ← {request[:44]}")
    except Exception as exc:
        print(f"❌ still failed: {type(exc).__name__}")

# %% [markdown]
# ### Nested schemas
#
# Structures compose. This is how you get an agent to return a *plan* — a list
# of typed steps that your code can then execute one by one.

# %%
class Step(BaseModel):
    """One step of a plan.

    search     = look something up (documents, the web, a database)
    calculate  = do arithmetic on numbers you already have
    write      = compose prose for the user
    ask_human  = the information is simply not available to you
    """

    number: int
    action: str = Field(description="a single concrete action, imperative mood")
    tool: Literal["search", "calculate", "write", "ask_human"]


class Plan(BaseModel):
    """A short plan for answering a complex question."""

    goal: str
    steps: list[Step] = Field(max_length=5)
    risks: list[str] = Field(default_factory=list, description="what could go wrong")


planner = llm.with_structured_output(Plan)
plan = planner.invoke(
    "Plan how to answer: 'Has our institute's travel budget kept pace with "
    "inflation since 2019, and what should we recommend?'"
)

print(f"GOAL: {plan.goal}\n")
for step in plan.steps:
    print(f"  {step.number}. [{step.tool:<9}] {step.action}")
print("\nRISKS:")
for risk in plan.risks:
    print("  -", risk)

# %% [markdown]
# Hold on to this cell. In notebook 10, the *first node* of our multi-agent
# system produces exactly such a `Plan`, and the graph then walks its steps.
# The "planner agent" that sounds so impressive is a Pydantic model and one
# `.invoke()`.

# %% [markdown]
# ## 4 · Reasoning models: thinking before speaking
#
# Now the second half. A **reasoning model** generates a private chain of
# thought before its answer. On Ollama we use `deepseek-r1:1.5b`; on the proxy,
# Qwen3.6 does the same thing when we enable it.
#
# Our helper hides the provider difference:
#
# ```python
# get_chat_model()          # thinking OFF -- fast, predictable (the default)
# get_chat_model(thinking=True)
# get_reasoning_model()     # thinking ON, bigger token budget
# ```

# %%
PUZZLE = (
    "A field station has 3 loggers. Logger A records every 5 minutes, "
    "B every 12 minutes, and C every 20 minutes. They all record together at "
    "08:00. At what time do all three next record together? "
    "Give the clock time."
)

print("=== WITHOUT thinking ===")
with timer("fast"):
    fast = get_chat_model(temperature=0, max_tokens=300).invoke(PUZZLE)
show(fast)

# %%
print("=== WITH thinking ===")
with timer("reasoning"):
    slow = get_reasoning_model(temperature=0, max_tokens=2500).invoke(PUZZLE)
show(slow)

# %% [markdown]
# (The answer is 09:00 — the least common multiple of 5, 12 and 20 is 60.)
#
# Scroll up through the "hidden thinking" block. That is not decoration: the
# model is genuinely using those tokens as scratch space, and it frequently
# catches its own arithmetic mistakes there.

# %% [markdown]
# ### Is it actually worth it? Measure.
#
# Reasoning is roughly 5–20× slower. Never pay that on faith — run your own
# mini-benchmark. Five questions is not science, but it is infinitely better
# than a vibe.

# %%
TASKS = [
    ("LCM", PUZZLE, "09:00"),
    ("arithmetic",
     "A grant is 18.4 million euros over 4 years for 214 people. "
     "What is the budget per person per year, in euros? Give a number.", "21495"),
    ("logic",
     "All loggers in Dept C are waterproof. Some waterproof loggers are solar. "
     "Does it follow that some Dept C loggers are solar? Answer yes or no.", "no"),
    ("counting",
     "How many times does the letter 'r' appear in 'interdisciplinary "
     "research infrastructure'? Give a number.", "7"),
    ("trap",
     "A researcher and a logger cost 1010 euros together. The researcher costs "
     "1000 euros more than the logger. How much is the logger? Give a number.", "5"),
]

# ⏱ Each reasoning call takes 30-90 seconds on a laptop, and there are two
# calls per task. Set QUICK = False for the full set when you have time -- the
# slowness is itself the point of this section.
QUICK = True
selected = TASKS[:3] if QUICK else TASKS
print(f"running {len(selected)} tasks x 2 modes = {len(selected) * 2} calls\n")

rows = []
for label, question, expected in selected:
    for mode, model in [
        ("fast", get_chat_model(temperature=0, max_tokens=400)),
        ("thinking", get_reasoning_model(temperature=0, max_tokens=3000)),
    ]:
        try:
            with timer(f"{label}/{mode}") as clock:
                reply = model.invoke(question + " Answer with the value only.")
            text = (reply.content or "").strip()

            # A thinking model that ran out of budget returns empty content with
            # a long chain of thought. Scoring that as "wrong" would be unfair --
            # it did the work, it just never got to write the answer down. We
            # grade the tail of its reasoning instead, and flag it.
            truncated = False
            if not text:
                thoughts = reply.additional_kwargs.get("reasoning_content") or ""
                text = thoughts.strip()[-200:]
                truncated = bool(text)
            normalised = text.lower().replace(",", "").replace(" ", "")
            rows.append({
                "task": label,
                "mode": mode,
                "seconds": round(clock.seconds, 1),
                "correct": expected.lower() in normalised,
                "cut off": truncated,
                "answer": text.replace("\n", " ")[-60:],
            })
        except Exception as exc:
            rows.append({"task": label, "mode": mode, "seconds": None,
                         "correct": None, "cut off": False,
                         "answer": f"ERROR {type(exc).__name__}"})

benchmark = pd.DataFrame(rows)
print()
print(benchmark.to_string(index=False))

# %%
summary = benchmark.groupby("mode").agg(
    accuracy=("correct", "mean"), mean_seconds=("seconds", "mean")
)
print(summary.round(2).to_string())
cut_off = benchmark["cut off"].sum()
if cut_off:
    print(f"\n⚠ {cut_off} answer(s) ran out of tokens mid-thought. That is not a\n"
          "  footnote -- it is the main operational risk of reasoning models:\n"
          "  you pay for all that thinking and can still get nothing back.")

print(
    "\nRead your own numbers, not the marketing. On a 2B-class model with these\n"
    "five tasks, thinking often costs 5-10x the latency for little or no gain:\n"
    "a small model's chain of thought is usually just a longer route to the\n"
    "same wrong answer. Reasoning pays off when the model is big enough for\n"
    "its own self-corrections to be worth something -- re-run this on the\n"
    "litellm backend and compare.\n\n"
    "The design lesson holds either way: think in the PLANNER and the CRITIC,\n"
    "never in the formatter or the router."
)

# %% [markdown]
# **Expect to see wrong answers in that table, in both modes.** The arithmetic
# task (18.4 M ÷ 214 ÷ 4 = 21,495) and the logic task ("no", the syllogism does
# not follow) both defeat a 2B model routinely. That is not a broken demo — it
# is the calibration you came for. A model that cannot divide three numbers
# reliably is a model that needs a calculator *tool*, which is exactly where
# notebook 07 goes next.

# %% [markdown]
# > **Note on the string matching above.** `"09:00" in answer` is a crude grader
# > and it will occasionally mark a right answer wrong. That is itself a lesson:
# > evaluating free-text output is hard, which is one more reason to make models
# > emit *structured* output when you intend to check it automatically.

# %% [markdown]
# ## 5 · The tension: structure vs. thinking
#
# Here is the friction nobody warns you about. Structured output works by
# constraining the model to emit one JSON object. Thinking works by letting the
# model ramble first. Ask for both at once and, depending on the provider, you
# get a parse error, an empty object, or thinking silently dropped.
#
# Try it — and note that *failing* here is the expected outcome on some setups:

# %%
class Verdict(BaseModel):
    """The result of working through a problem."""

    answer: str = Field(
        description="ONLY the final value, at most 20 characters, e.g. '09:00' "
                    "or '391'. No working, no explanation, no units unless asked."
    )
    method: str = Field(description="how you got there, ONE short sentence")
    confident: bool


try:
    with timer("structured + thinking"):
        result = get_reasoning_model(temperature=0, max_tokens=2500).with_structured_output(
            Verdict
        ).invoke(PUZZLE)
    print(result.model_dump())
except Exception as exc:
    print(f"❌ {type(exc).__name__}: {str(exc)[:200]}")
    print("   ^ this is the failure mode. See the robust pattern below.")

# %% [markdown]
# ### The pattern that always works: two calls
#
# **Think in prose, then structure in a second, cheap call.** It is more robust
# than any single-call trick, it is easy to debug (you can read the reasoning),
# and the second call is fast because it only has to reformat.
#
# This "reason, then format" split is used by essentially every production agent
# framework, including the ones we build in notebooks 09 and 10.

# %%
def reason_then_structure(question: str, schema: type[BaseModel]):
    """Step 1: let the model think freely. Step 2: extract the fields."""
    thought = get_reasoning_model(temperature=0, max_tokens=3000).invoke(
        f"{question}\n\nWork through this carefully."
    )
    prose = thought.content or thought.additional_kwargs.get("reasoning_content", "")

    # Hand the formatter only the CONCLUSION. Give it the whole chain of thought
    # and a small model will dutifully copy all of it into the first string
    # field, then run out of tokens before it reaches the others.
    conclusion = prose.strip()[-1200:]

    formatter = get_chat_model(temperature=0, max_tokens=600).with_structured_output(
        schema
    )
    extracted = formatter.invoke(
        "Below is someone's worked solution. Extract the requested fields.\n"
        "Copy nothing but the values -- do NOT repeat the working.\n\n"
        f"WORKED SOLUTION:\n{conclusion}"
    )
    return extracted, prose


try:
    with timer("reason-then-structure"):
        verdict, prose = reason_then_structure(PUZZLE, Verdict)

    print("reasoning was", len(prose), "characters long\n")
    for field, value in verdict.model_dump().items():
        print(f"  {field:<10} = {value!r}")
except Exception as exc:
    print(f"❌ {type(exc).__name__}: {str(exc)[:200]}")
    print("\n   Even the two-call pattern is not bulletproof on a 2B model.")
    print("   In production you would add .with_retry() here -- one re-ask fixes")
    print("   the large majority of these, at the cost of one extra call.")

# %% [markdown]
# ## What you now have
#
# | Capability | Notebook | Why an agent needs it |
# |------------|----------|-----------------------|
# | call a model | 02 | the engine |
# | embeddings & similarity | 03 | measure meaning |
# | projection & clustering | 04 | cheap routing |
# | retrieval | 05 | knowledge beyond training data |
# | **structured output** | **06** | **decisions a program can act on** |
# | **reasoning** | **06** | **multi-step problems** |
#
# One piece is missing: the model can only *talk*. It cannot look anything up,
# run a calculation, or touch the world. That is **tools** — and it is where
# agents actually begin.
#
# → Next: **07 · Tools and your first agent**

# %% [markdown]
# ## Your turn — 10 minutes
#
# 1. Write a Pydantic schema for something in your own work (a paper's metadata,
#    an experiment's parameters, a participant record) and extract it from a
#    paragraph of real text. How does it fail?
# 2. Add `Field(description=...)` to a field the model keeps getting wrong. The
#    description is a prompt — use it as one.
# 3. Add two tasks from your field to `TASKS`. Does thinking help *there*?
# 4. Make `Routing.confidence` meaningful: find a request where the model
#    correctly reports low confidence. What should an agent do with that number?
