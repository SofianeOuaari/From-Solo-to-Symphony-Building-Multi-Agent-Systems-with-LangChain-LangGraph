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
# # 10 · Capstone — the Research Desk
#
# Everything from the previous nine notebooks, in one system.
#
# We are building a **research desk**: you ask it a question about the institute,
# and a team of agents plans, researches, computes, drafts, critiques, revises,
# and then asks a human before it ships.
#
# ```
#  question
#     │
#     ▼
#  ┌────────┐   a typed Plan (nb 06)
#  │ PLANNER│
#  └───┬────┘
#      ▼
#  ┌──────────┐  retrieval as a tool (nb 05, 07)
#  │RESEARCHER│
#  └───┬──────┘
#      ▼
#  ┌────────┐   calculator tool (nb 07)
#  │ ANALYST│
#  └───┬────┘
#      ▼
#  ┌────────┐ ◄──────────────┐
#  │ WRITER │                │ revise (max 2)
#  └───┬────┘                │
#      ▼                     │
#  ┌────────┐                │
#  │ CRITIC │────────────────┘   a cycle (nb 08)
#  └───┬────┘
#      ▼ approved
#  ┌─────────┐
#  │  HUMAN  │  interrupt() (nb 08)
#  └───┬─────┘
#      ▼
#   delivered
# ```
#
# Which notebook each piece comes from:
#
# | Component | From |
# |-----------|------|
# | model access, both backends | 02 |
# | embeddings | 03 |
# | chunking + vector store + retrieval | 05 |
# | `Plan` and `Review` as Pydantic schemas | 06 |
# | tools and the agent loop | 07 |
# | `StateGraph`, reducers, cycles, checkpointer, `interrupt` | 08 |
# | roles, supervision, traces, step budgets | 09 |

# %%
# %pip install -q -r ../requirements.txt

# %%
import pathlib
import sys

ROOT = next(
    p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir()
)
sys.path.insert(0, str(ROOT))

import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from common.workshop_setup import describe_model, draw, get_chat_model, get_embeddings

# Two models, on purpose: a careful one for judgement, a fast one for
# reformatting. In production this is where most of your cost savings live.
thinker = get_chat_model(temperature=0.0, max_tokens=900)
scribe = get_chat_model(temperature=0.3, max_tokens=700)

print("judgement :", describe_model(thinker))
print("writing   :", describe_model(scribe))

# %% [markdown]
# ## The knowledge base and the tools

# %%
handbook = (ROOT / "05_rag_basics" / "knowledge_base.md").read_text()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100, separators=["\n## ", "\n\n", "\n", ". ", " "]
).split_text(handbook)

vector_store = InMemoryVectorStore(get_embeddings("nomic-embed-text:latest"))
vector_store.add_texts(chunks)

# A tiny audit log, so we can show *what was actually retrieved*. Provenance is
# not a nice-to-have: without it nobody can check the system's claims.
RETRIEVAL_LOG: list[dict] = []


@tool
def search_handbook(query: str) -> str:
    """Search the internal handbook of the Institute for Adaptive Systems.

    Covers departments and who leads them, the HELIOS compute cluster, data
    management and DPA-2024, publication and open access, travel budgets,
    the PhD programme, and ethics approval.
    Pass a full descriptive question, not a single keyword.
    """
    hits = vector_store.similarity_search_with_score(query, k=3)
    RETRIEVAL_LOG.append(
        {"query": query, "scores": [round(float(s), 3) for _, s in hits]}
    )
    return "\n\n---\n\n".join(doc.page_content for doc, _ in hits)


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "96 * 4" or "18400000 / (214 * 4)".
    Use this for ALL arithmetic. Numbers and + - * / ** % ( ) only."""
    if not set(expression) <= set("0123456789.+-*/%()e "):
        return f"Error: forbidden characters in {expression!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"


print(f"indexed {len(chunks)} chunks")

# %% [markdown]
# ## The shared state
#
# One dictionary, flowing through the whole team. Note which fields accumulate
# (`Annotated[..., operator.add]`) and which get overwritten — that choice is
# where subtle multi-agent bugs live.

# %%
class Step(BaseModel):
    """One step of the plan.

    handbook   = this needs looking up in the institute handbook
    arithmetic = this needs numbers combined (÷, ×, %) that we already have
    neither    = this needs neither, e.g. summarising or rephrasing
    """

    action: str = Field(description="one concrete sub-question, imperative mood")
    needs: Literal["handbook", "arithmetic", "neither"] = Field(
        description="Pick 'arithmetic' ONLY if a calculation is genuinely "
                    "required. Looking a figure up is 'handbook', not arithmetic."
    )


class Plan(BaseModel):
    """How to answer the question."""

    restated: str = Field(
        description="ONE sentence restating the user's question. Restate only -- "
                    "do not answer it, and do not add any facts of your own."
    )
    steps: list[Step] = Field(max_length=4)
    answerable: bool = Field(
        description="false if the handbook is clearly the wrong source for this"
    )


class Review(BaseModel):
    """A critic's verdict on a draft."""

    approved: bool = Field(
        description="true only if every claim is supported by the evidence "
                    "and the question is fully answered"
    )
    problems: list[str] = Field(
        default_factory=list, description="what must be fixed; max 3 items, short"
    )


class DeskState(TypedDict):
    question: str
    # A plain dict, not the Plan object. Graph state is written to the
    # checkpointer, so keep it JSON-serialisable -- stuffing arbitrary Python
    # objects in there works until the day you swap InMemorySaver for Postgres.
    plan: dict | None
    evidence: Annotated[list[str], operator.add]
    numbers: str
    draft: str
    problems: list[str]
    approved: bool
    revisions: int
    delivered: bool
    trace: Annotated[list[str], operator.add]

# %% [markdown]
# ## The agents
#
# Each one is a system prompt, a tool menu, and a narrow contract. Read the
# prompts: every line of them exists because of a failure mode we met earlier.

# %%
research_agent = create_agent(
    model=thinker,
    tools=[search_handbook],
    system_prompt=(
        "You are a research librarian at the Institute for Adaptive Systems.\n"
        "- ALWAYS search the handbook before answering. Search more than once "
        "  if the question has several parts.\n"
        "- Report ONLY what the handbook says, as short bullet points, each with "
        "  the exact figure or name.\n"
        "- If the handbook does not cover something, write "
        "  'NOT IN HANDBOOK: <what was missing>'.\n"
        "- Never speculate, never fill gaps from general knowledge."
    ),
)

analysis_agent = create_agent(
    model=thinker,
    tools=[calculate],
    system_prompt=(
        "You are a quantitative analyst.\n"
        "- You receive evidence gathered by a librarian.\n"
        "- Use the calculator for EVERY arithmetic operation. Never compute "
        "  mentally.\n"
        "- Output at most 4 bullet points: the derived number and what it means.\n"
        "- If no arithmetic is needed, reply exactly 'No computation needed.'"
    ),
)

# %% [markdown]
# ## The nodes

# %%
def plan_node(state: DeskState) -> dict:
    plan = thinker.with_structured_output(Plan).invoke(
        "You are the head of a research desk that answers questions using the "
        "internal handbook of the Institute for Adaptive Systems (departments, "
        "HELIOS cluster, data policy, publications, travel, PhD programme, "
        "ethics).\n\nMake a short plan for this question:\n\n"
        f"{state['question']}"
    )
    return {
        "plan": plan.model_dump(),
        "trace": [f"planner: {len(plan.steps)} steps, answerable={plan.answerable}"],
    }


def research_node(state: DeskState) -> dict:
    plan = state["plan"]
    tasks = [s["action"] for s in plan["steps"] if s["needs"] in ("handbook", "neither")]
    brief = "\n".join(f"- {t}" for t in tasks) or f"- {state['question']}"

    out = research_agent.invoke(
        {"messages": [HumanMessage(
            f"Question: {plan['restated']}\n\nFind evidence for each point:\n{brief}"
        )]}
    )
    findings = out["messages"][-1].content
    searches = sum(
        len(getattr(m, "tool_calls", []) or []) for m in out["messages"]
    )
    return {"evidence": [findings], "trace": [f"researcher: {searches} search(es)"]}


def analysis_node(state: DeskState) -> dict:
    wanted = any(s["needs"] == "arithmetic" for s in state["plan"]["steps"])

    # The planner is a small model and mislabels steps. A second, cheap guard:
    # if the librarian found nothing, there is nothing to compute either.
    evidence = "\n".join(state["evidence"])
    nothing_found = "NOT IN HANDBOOK" in evidence.upper()

    if not wanted or nothing_found:
        reason = "nothing to compute" if not wanted else "no evidence to compute from"
        return {"numbers": "No computation needed.", "trace": [f"analyst: skipped ({reason})"]}

    out = analysis_agent.invoke(
        {"messages": [HumanMessage(
            f"Question: {state['question']}\n\nEvidence:\n" + "\n".join(state["evidence"])
        )]}
    )
    return {"numbers": out["messages"][-1].content, "trace": ["analyst: computed"]}


def write_node(state: DeskState) -> dict:
    evidence = "\n".join(state["evidence"])
    if state["revisions"] == 0:
        task = (
            f"Question: {state['question']}\n\n"
            f"EVIDENCE:\n{evidence}\n\nDERIVED NUMBERS:\n{state['numbers']}\n\n"
            "Write the answer."
        )
    else:
        task = (
            f"Question: {state['question']}\n\n"
            f"EVIDENCE:\n{evidence}\n\nDERIVED NUMBERS:\n{state['numbers']}\n\n"
            f"YOUR PREVIOUS DRAFT:\n{state['draft']}\n\n"
            "The reviewer rejected it for these reasons:\n"
            + "\n".join(f"- {p}" for p in state["problems"])
            + "\n\nRewrite it, fixing every point."
        )

    draft = scribe.invoke([
        SystemMessage(
            "You are a science writer. Write at most 120 words of plain prose for "
            "a busy professor. Keep every figure exactly as given. Use ONLY the "
            "evidence provided -- if it says NOT IN HANDBOOK, say so plainly in "
            "the answer. No preamble, no bullet points, no invented detail."
        ),
        HumanMessage(task),
    ]).content

    return {
        "draft": draft,
        "revisions": state["revisions"] + 1,
        "trace": [f"writer: draft {state['revisions'] + 1}"],
    }


WORD_LIMIT = 130


def critic_node(state: DeskState) -> dict:
    """Two kinds of check, deliberately separated.

    Anything a computer can decide -- length, presence of a required token --
    is decided *in code*. The model is asked only for the judgement a computer
    cannot make: is every claim actually supported by the evidence?

    We learned this the hard way: asked to count words, a small model will
    confidently report "136 words" about a 60-word draft. Never delegate to an
    LLM something `len(text.split())` already answers.
    """
    mechanical: list[str] = []
    words = len(state["draft"].split())
    if words > WORD_LIMIT:
        mechanical.append(f"Too long: {words} words, limit is {WORD_LIMIT}.")

    review = thinker.with_structured_output(Review).invoke(
        "You are a reviewer checking a draft against its evidence.\n"
        "Reject ONLY if you can point to a concrete problem:\n"
        "  - the draft states a fact that is not in the evidence\n"
        "  - the draft drops or changes a figure that is in the evidence\n"
        "  - the evidence says NOT IN HANDBOOK but the draft answers anyway\n"
        "  - the draft does not address the question\n"
        "Do NOT judge length, tone or style -- those are handled elsewhere.\n"
        "If you find no such problem, approve it.\n\n"
        f"QUESTION: {state['question']}\n\n"
        f"EVIDENCE:\n{chr(10).join(state['evidence'])}\n\n"
        f"NUMBERS:\n{state['numbers']}\n\n"
        f"DRAFT:\n{state['draft']}"
    )

    problems = mechanical + list(review.problems)
    approved = review.approved and not mechanical
    verdict = "approved" if approved else f"rejected ({len(problems)})"
    return {
        "approved": approved,
        "problems": problems,
        "trace": [f"critic: {verdict} [{words} words]"],
    }


MAX_REVISIONS = 2


def after_critic(state: DeskState) -> str:
    """The exit condition for the revision cycle."""
    if state["approved"]:
        return "human"
    if state["revisions"] >= MAX_REVISIONS:
        return "human"          # ship it with the caveat rather than loop forever
    return "revise"


def human_node(state: DeskState) -> dict:
    """The gate. Nothing leaves the desk without a person seeing it."""
    decision = interrupt({
        "question": state["question"],
        "draft": state["draft"],
        "critic_approved": state["approved"],
        "outstanding_problems": state["problems"],
        "prompt": "Reply 'approve' to deliver, or give revision instructions.",
    })

    if str(decision).strip().lower() == "approve":
        return {"delivered": True, "trace": ["human: approved"]}

    # Any other reply is treated as feedback and sent back to the writer.
    return {
        "delivered": False,
        "approved": False,
        "problems": [f"Human reviewer: {decision}"],
        "revisions": 0,                      # a human's request resets the budget
        "trace": [f"human: requested changes — {decision}"],
    }


def after_human(state: DeskState) -> str:
    return "done" if state["delivered"] else "revise"

# %% [markdown]
# ## Wiring the graph
#
# Notice that the *only* thing that makes this a "multi-agent system" rather
# than a script is this wiring plus the shared state. There is no hidden
# machinery.

# %%
builder = StateGraph(DeskState)
builder.add_node("planner", plan_node)
builder.add_node("researcher", research_node)
builder.add_node("analyst", analysis_node)
builder.add_node("writer", write_node)
builder.add_node("critic", critic_node)
builder.add_node("human", human_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "analyst")
builder.add_edge("analyst", "writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges("critic", after_critic, {"revise": "writer", "human": "human"})
builder.add_conditional_edges("human", after_human, {"revise": "writer", "done": END})

# The checkpointer is what makes interrupt() resumable.
desk = builder.compile(checkpointer=InMemorySaver())
draw(desk)

# %% [markdown]
# ## Running it
#
# The helper below streams the graph and prints a transcript. Then it pauses at
# the human gate, exactly as designed.

# %%
ICONS = {"planner": "🗂️ ", "researcher": "📚", "analyst": "🧮",
         "writer": "✍️ ", "critic": "🧐", "human": "🙋"}


def start(question: str, thread_id: str):
    RETRIEVAL_LOG.clear()
    started = time.perf_counter()
    # recursion_limit is a top-level config key, not a separate argument.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}

    print(f"\n{'=' * 76}\n❓ {question}\n{'=' * 76}")

    initial = {
        "question": question, "plan": None, "evidence": [], "numbers": "",
        "draft": "", "problems": [], "approved": False, "revisions": 0,
        "delivered": False, "trace": [],
    }

    for update in desk.stream(initial, config):
        if "__interrupt__" in update:
            continue
        for node, output in update.items():
            icon = ICONS.get(node, "  ")
            if node == "planner":
                plan = output["plan"]
                print(f"\n{icon} PLANNER — {plan['restated']}")
                for i, step in enumerate(plan["steps"], 1):
                    print(f"     {i}. [{step['needs']:<10}] {step['action']}")
            elif node == "researcher":
                print(f"\n{icon} RESEARCHER\n{_indent(output['evidence'][0])}")
                for entry in RETRIEVAL_LOG:
                    print(f"     🔎 {entry['query'][:64]!r} → {entry['scores']}")
            elif node == "analyst":
                print(f"\n{icon} ANALYST\n{_indent(output['numbers'])}")
            elif node == "writer":
                print(f"\n{icon} WRITER (draft {output['revisions']})\n"
                      f"{_indent(output['draft'])}")
            elif node == "critic":
                if output["approved"]:
                    print(f"\n{icon} CRITIC — ✅ approved")
                else:
                    print(f"\n{icon} CRITIC — ❌ rejected")
                    for problem in output["problems"]:
                        print(f"     · {problem}")

    print(f"\n[elapsed {time.perf_counter() - started:.1f}s]")
    return config


def _indent(text: str, width: int = 900) -> str:
    body = str(text).strip()[:width]
    return "\n".join("     " + line for line in body.splitlines())


def show_gate(config):
    """Print whatever the graph is waiting for."""
    state = desk.get_state(config)
    if not state.tasks or not state.tasks[0].interrupts:
        print("(graph is not paused)")
        return None
    payload = state.tasks[0].interrupts[0].value
    print(f"\n{'─' * 76}\n🙋 WAITING FOR A HUMAN\n{'─' * 76}")
    print("critic approved:", payload["critic_approved"])
    if payload["outstanding_problems"]:
        print("unresolved:", payload["outstanding_problems"])
    print(f"\nDRAFT:\n{payload['draft'].strip()}")
    print(f"\n> {payload['prompt']}")
    return payload


def resume(config, decision: str):
    print(f"\n[human says: {decision!r}]")
    for update in desk.stream(Command(resume=decision), config):
        if "__interrupt__" in update:
            continue
        for node, output in update.items():
            icon = ICONS.get(node, "  ")
            if node == "writer":
                print(f"\n{icon} WRITER (revision)\n{_indent(output['draft'])}")
            elif node == "critic":
                print(f"\n{icon} CRITIC — {'✅' if output['approved'] else '❌'}")
            elif node == "human":
                print(f"\n{icon} {output['trace'][0]}")

    final = desk.get_state(config).values
    if final.get("delivered"):
        print(f"\n{'=' * 76}\n📬 DELIVERED\n{'=' * 76}\n{final['draft'].strip()}")
    return final

# %% [markdown]
# ### Run 1 — a question needing research *and* arithmetic

# %%
config = start(
    "What is the total number of GPUs across HELIOS, and how much GPU memory "
    "would one node's worth represent as a share of the whole cluster?",
    thread_id="run-1",
)

# %%
show_gate(config)

# %%
final = resume(config, "approve")

# %% [markdown]
# ### Run 2 — the human asks for changes
#
# This is the path that matters in practice. The human is not a rubber stamp;
# their feedback re-enters the graph as a revision instruction and the writer
# runs again.

# %%
config2 = start(
    "I'm a first-year PhD student. What do I have to get done in year one?",
    thread_id="run-2",
)

# %%
show_gate(config2)

# %%
resume(config2, "Too formal. Rewrite it as a friendly checklist a student can tick off.")

# %% [markdown]
# The revised draft comes back to the gate, because the human node is on the
# path every time. Approve it to finish the run:

# %%
resume(config2, "approve")

# %% [markdown]
# ### Run 3 — a question the handbook cannot answer
#
# The single most valuable behaviour in the whole system. Watch the chain:
# the researcher reports `NOT IN HANDBOOK`, the critic is instructed to reject
# any draft that answers anyway, and the human sees an honest non-answer.
#
# A system that says "I don't know" is usable. A system that guesses is not.

# %%
config3 = start(
    "What is the institute's policy on using generative AI in manuscripts?",
    thread_id="run-3",
)

# %%
show_gate(config3)

# %%
resume(config3, "approve")

# %% [markdown]
# ### A caveat on the critic, and it is an important one
#
# On a 2B model you will sometimes watch the critic **approve a draft that
# contradicts itself** — e.g. a sentence saying "256 GB per node" and then
# "roughly 512 GB" a clause later. The critic is a language model, and a small
# one catches obvious unsupported claims while missing subtle internal
# inconsistency.
#
# So be precise about what a critic buys you: it is a **filter that raises the
# floor**, not a guarantee. It is also the cheapest place in the whole system to
# spend a bigger model. Try it:
#
# ```python
# thinker = get_chat_model(backend="litellm", temperature=0.0, max_tokens=900)
# ```
#
# A 35B critic over 2B workers is a very good trade: one expensive call per
# revision, and it catches things the workers cannot see in themselves.

# %% [markdown]
# ### Why the critic does not count words
#
# The first version of this notebook asked the critic to reject drafts over 120
# words. It rejected **everything**, reporting things like *"136 words, limit
# 120"* about a 58-word paragraph. Small models cannot count.
#
# The fix is a rule worth carrying out of this workshop:
#
# > **Put deterministic checks in code and judgement in the model.**
# > If `len(text.split())` can answer it, never ask an LLM.
#
# Look at `critic_node`: the length check is three lines of Python, and the
# model is asked only the thing Python cannot decide — *is this claim actually
# supported by the evidence?* Mixing the two is the single most common way a
# critic agent becomes useless: it burns your revision budget on imaginary
# problems and never gets to the real ones.

# %% [markdown]
# ## Observability: what did that actually cost?
#
# You cannot improve what you do not measure. Even this crude accounting tells
# you where the time went — and where to put a smaller model.

# %%
for thread in ["run-1", "run-2", "run-3"]:
    values = desk.get_state({"configurable": {"thread_id": thread}}).values
    if not values:
        continue
    print(f"\n{thread}: delivered={values.get('delivered')} "
          f"revisions={values.get('revisions')}")
    for line in values.get("trace", []):
        print("   ·", line)

# %% [markdown]
# For anything beyond a workshop, use real tracing rather than print statements:
#
# ```bash
# export LANGSMITH_TRACING=true
# export LANGSMITH_API_KEY=...
# ```
#
# Every node, every LLM call, every token and every tool result then shows up in
# a waterfall view. With a six-agent system and a cycle, this stops being a
# luxury within about a day.

# %% [markdown]
# ## What to take away
#
# **1. There is no "agent" abstraction.** An agent is a loop around a model that
# can call tools. A multi-agent system is a graph of those loops over shared
# state. You wrote both by hand in notebooks 07 and 08 before any framework was
# involved — keep that picture.
#
# **2. Roles are prompts, and prompts are specifications.** Every line in the
# researcher's system prompt exists because of a failure. That is how those
# prompts should grow: from observed failures, not from imagination.
#
# **3. Structure at the boundaries.** Agents talk to each other through Pydantic
# models and typed state, not prose. Prose between agents is how errors travel.
#
# **4. Every cycle needs a budget, every system needs a trace, every consequential
# action needs a human.** These three are not polish. They are the difference
# between a demo and something you would let near real work.
#
# **5. The hard part is not the code.** It is deciding what the roles are, who
# needs to see what, and where the system is allowed to be wrong. That is
# research design, and it is your job, not the framework's.
#
# ## Where to go next
#
# * **Swap the knowledge base for your own.** Point the splitter at your papers
#   or your lab wiki. Everything else works unchanged. This is the single most
#   valuable hour you can spend after today.
# * **Add a real tool** — your own API, a database, a simulation.
# * **Parallelise.** Have the planner emit N sub-questions and fan them out to N
#   researchers at once (LangGraph's `Send` API).
# * **Evaluate.** Write 20 questions with known answers and measure. Without
#   this you are guessing, and every change will feel like an improvement.
# * **Read the docs**: <https://langchain-ai.github.io/langgraph/>

# %% [markdown]
# ## Your turn — the rest of the session
#
# Pick one and actually build it:
#
# 1. **A second knowledge source.** Add a `search_papers` tool over a different
#    corpus. Now the researcher must choose *which* source — does it choose well?
# 2. **A fact-checker agent** that takes the final draft and verifies each claim
#    against the handbook independently, then reports a per-claim verdict.
# 3. **Parallel researchers.** Fan the plan's steps out to one researcher each
#    and merge the evidence with a reducer. Measure the speed-up.
# 4. **Your own domain.** Replace the handbook with something from your own
#    work and adjust only the prompts. Report back what broke — that list is
#    the most interesting output of this workshop.
