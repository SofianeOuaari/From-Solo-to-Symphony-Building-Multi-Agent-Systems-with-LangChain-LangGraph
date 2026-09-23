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
# # 09 · Multi-agent collaboration
#
# We have all the parts: a model (02), embeddings (03, 04), retrieval (05),
# structured decisions (06), tools and the agent loop (07), and a graph with
# state, cycles and memory (08).
#
# Now the actual subject of the workshop: **several agents, with different
# roles, working on one task**.
#
# ## First, the honest question: why bother?
#
# One agent with fifteen tools is simpler, cheaper and easier to debug than five
# agents with three tools each. So multi-agent is only worth it for specific
# reasons:
#
# | Reason | What it buys you |
# |--------|------------------|
# | **Tool selection accuracy** | past ~10–15 tools, one agent starts picking wrong. Five menus of three are easy. |
# | **Context isolation** | the writer never sees 40 retrieved chunks; it sees the analyst's summary. Shorter prompts, better answers, less cost. |
# | **Role clarity** | "be a harsh critic" and "be an enthusiastic drafter" in one system prompt fight each other. In two agents they don't. |
# | **Different models per role** | a slow reasoning model to plan, a fast cheap one to format. |
# | **Parallelism** | three independent literature searches at once. |
# | **Separable evaluation** | you can test the retriever without the writer. |
#
# And the costs, which are real:
#
# * every handoff is extra latency and extra tokens
# * failures compound — 4 agents at 90% reliability each is 66% end-to-end
# * debugging gets harder; you need traces, which is why we print them
# * agents can loop, disagree, or politely hand work back and forth forever
#
# **Rule of thumb: start with one agent. Split it when you can name which of the
# reasons above applies.**

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
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.workshop_setup import describe_model, draw, get_chat_model, get_embeddings

llm = get_chat_model(temperature=0, max_tokens=700)
print("using:", describe_model(llm))

# %% [markdown]
# ## Setting up the shared resources
#
# The handbook from notebook 05 becomes our knowledge base, wrapped as a tool.

# %%
handbook = (ROOT / "05_rag_basics" / "knowledge_base.md").read_text()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100, separators=["\n## ", "\n\n", "\n", ". ", " "]
).split_text(handbook)

vector_store = InMemoryVectorStore(get_embeddings("nomic-embed-text:latest"))
vector_store.add_texts(chunks)


@tool
def search_handbook(query: str) -> str:
    """Search the internal handbook of the Institute for Adaptive Systems.

    Covers departments and their leaders, the HELIOS cluster, data management
    and DPA-2024, publication and open access, travel budgets, the PhD
    programme, and ethics approval. Pass a full question, not a keyword.
    """
    hits = vector_store.similarity_search(query, k=3)
    return "\n\n---\n\n".join(h.page_content for h in hits)


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "96 * 4" or "18400000 / 214".
    Use this for ALL arithmetic. Numbers and + - * / ** % ( ) only."""
    if not set(expression) <= set("0123456789.+-*/%()e "):
        return f"Error: forbidden characters in {expression!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error: {exc}"


print(f"knowledge base: {len(chunks)} chunks")

# %% [markdown]
# ## Pattern 1 · Sequential pipeline
#
# The simplest form of collaboration: A finishes, hands to B, B hands to C. No
# agent decides anything about the flow — *you* decided it when you drew the
# graph.
#
# Not glamorous, but often the right answer. Use it when the steps genuinely
# always happen in the same order.
#
# ```
# researcher ──► analyst ──► writer
# ```

# %%
class PipelineState(TypedDict):
    question: str
    findings: str
    analysis: str
    report: str
    trace: Annotated[list[str], operator.add]


def make_role(name: str, persona: str, tools: list | None = None):
    """Create one agent. A 'role' is a system prompt plus a tool menu."""
    from langchain.agents import create_agent

    if tools:
        agent = create_agent(model=llm, tools=tools, system_prompt=persona)

        def run(task: str) -> str:
            out = agent.invoke({"messages": [HumanMessage(task)]})
            return out["messages"][-1].content

    else:
        def run(task: str) -> str:
            return llm.invoke([SystemMessage(persona), HumanMessage(task)]).content

    run.role_name = name
    return run


researcher = make_role(
    "Researcher",
    "You are a research librarian. Search the handbook and report ONLY what it "
    "says, as short bullet points with the exact numbers. Never speculate. "
    "If the handbook is silent, say 'not in the handbook'.",
    tools=[search_handbook],
)

analyst = make_role(
    "Analyst",
    "You are a quantitative analyst. You are given findings. Do any arithmetic "
    "with the calculator tool -- never in your head. Output 2-4 bullet points "
    "of derived numbers and what they imply.",
    tools=[calculate],
)

writer = make_role(
    "Writer",
    "You are a science writer. Turn the material you are given into a clear "
    "answer of at most 120 words for a busy professor. Keep every number. "
    "No preamble, no bullet points -- prose.",
)

# %%
def research_node(state: PipelineState) -> dict:
    findings = researcher(f"Find everything relevant to: {state['question']}")
    return {"findings": findings, "trace": [f"Researcher: {len(findings)} chars"]}


def analysis_node(state: PipelineState) -> dict:
    analysis = analyst(
        f"Question: {state['question']}\n\nFindings:\n{state['findings']}"
    )
    return {"analysis": analysis, "trace": [f"Analyst: {len(analysis)} chars"]}


def writing_node(state: PipelineState) -> dict:
    report = writer(
        f"Question: {state['question']}\n\n"
        f"Findings:\n{state['findings']}\n\nAnalysis:\n{state['analysis']}"
    )
    return {"report": report, "trace": ["Writer: done"]}


builder = StateGraph(PipelineState)
builder.add_node("researcher", research_node)
builder.add_node("analyst", analysis_node)
builder.add_node("writer", writing_node)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "analyst")
builder.add_edge("analyst", "writer")
builder.add_edge("writer", END)

pipeline = builder.compile()
draw(pipeline)

# %%
QUESTION = "How much total GPU capacity does HELIOS have, and how much RAM per node?"

result = pipeline.invoke(
    {"question": QUESTION, "findings": "", "analysis": "", "report": "", "trace": []}
)

for stage, content in [
    ("🔍 RESEARCHER", result["findings"]),
    ("📊 ANALYST", result["analysis"]),
    ("✍️  WRITER", result["report"]),
]:
    print(f"\n{'─' * 72}\n{stage}\n{'─' * 72}\n{content.strip()[:700]}")

# %% [markdown]
# Notice the **context isolation** in action: the writer never saw a single
# retrieved chunk. It saw a short findings list and a short analysis. Its prompt
# is a fraction of the size, and its job is correspondingly easier.
#
# But this pipeline is dumb in one specific way: it *always* runs all three
# agents. Ask it "what's 2+2" and the librarian still searches the handbook.

# %% [markdown]
# ## Pattern 2 · Supervisor — the workhorse
#
# Now let an agent decide. A **supervisor** reads the shared conversation and
# picks who works next, over and over, until it decides the task is done.
#
# ```
#            ┌──────────────┐
#       ┌───►│  SUPERVISOR  │────► END
#       │    └──────┬───────┘
#       │           │ picks one
#       │   ┌───────┼────────┐
#       │   ▼       ▼        ▼
#       │ librarian analyst writer
#       └───────────┴────────┘
#             report back
# ```
#
# This is the most widely used multi-agent topology, because:
#
# * the routing logic lives in **one** place you can read and test
# * adding a specialist means adding a node and a line to its description
# * the supervisor can call the same worker twice, or skip workers entirely
#
# Its weakness: the supervisor is a bottleneck, and every hop costs an LLM call.

# %%
WORKERS = {
    "librarian": "Searches the institute handbook. Use for any factual question "
                 "about the institute, its policies, people, or infrastructure.",
    "analyst": "Does arithmetic and derives quantities. Use when numbers must be "
               "computed, compared, or divided.",
    "writer": "Writes the final polished answer for the user. Use LAST, once the "
              "facts and numbers are gathered.",
}


class Decision(BaseModel):
    """The supervisor's choice of who works next."""

    next_worker: Literal["librarian", "analyst", "writer", "FINISH"]
    instruction: str = Field(
        description="a single concrete instruction for that worker, max 30 words; "
                    "empty if FINISH"
    )


class TeamState(TypedDict):
    question: str
    # Everything every agent has contributed -- the shared blackboard.
    scratchpad: Annotated[list[str], operator.add]
    # Who has already worked. The supervisor is a small model and will happily
    # forget; we make it impossible to forget by putting it in the state.
    used: Annotated[list[str], operator.add]
    final_answer: str
    next_worker: str
    instruction: str
    steps: int


SUPERVISOR_PROMPT = """You are the supervisor of a small research team.

Your team:
{team}

The user's question:
{question}

Workers already used (and how many times): {used}
Only the workers listed above are still available to you.

Work done so far:
{scratchpad}

Choose the ONE worker who should act next, and give them a single concrete
instruction. Rules:
- Any question about the institute MUST go to the librarian first. Never let
  the writer state an institutional fact the librarian has not confirmed.
- If numbers still need computing, use the analyst.
- Use the writer only ONCE, at the end, when the facts and numbers are in.
- Reply FINISH as soon as the writer has produced an answer.
- Never send a worker who has already worked, unless something genuinely new
  needs looking up."""


from collections import Counter

MAX_STEPS = 6          # total supervisor turns
MAX_PER_WORKER = 2     # how often any one worker may be called


def supervisor_node(state: TeamState) -> dict:
    """Pick the next worker.

    Three of the four decisions here are made in plain Python. Each guard was
    added because we *watched the system fail without it* -- which is exactly
    how you should grow your own:

      1. no step budget      → the supervisor dithers forever
      2. no writer guard     → it calls the writer 5x with the same instruction
      3. no per-worker cap   → it calls the librarian 5x, each time getting the
                               same "not in the handbook" answer back

    All three are *facts about what has happened*. None of them is a judgement.
    Facts belong in code; only the choice of who is useful next goes to the model.
    """
    counts = Counter(state["used"])
    step = state["steps"] + 1

    def finish(note: str) -> dict:
        return {"next_worker": "FINISH", "instruction": "", "steps": step,
                "scratchpad": [f"[supervisor] {note}"]}

    # Guard 1: a hard total budget.
    if state["steps"] >= MAX_STEPS:
        return finish("step budget exhausted → FINISH")

    # Guard 2: the writer is terminal. Once there is an answer, we are done.
    if state.get("final_answer"):
        return finish("the writer has answered → FINISH")

    # Guard 3: a worker who has had its turns is off the menu.
    available = [w for w in WORKERS if counts[w] < MAX_PER_WORKER]
    if not available:
        return finish("every worker is used up → FINISH")

    scratchpad = "\n\n".join(state["scratchpad"]) or "(nothing yet)"
    team = "\n".join(f"- {name}: {WORKERS[name]}" for name in available)
    used = ", ".join(f"{name} x{n}" for name, n in counts.items()) or "nobody yet"

    decision = llm.with_structured_output(Decision).invoke(
        SUPERVISOR_PROMPT.format(team=team, question=state["question"],
                                 scratchpad=scratchpad[:3000], used=used)
    )

    chosen, instruction = decision.next_worker, decision.instruction

    # ...and if it picks one anyway, we quietly correct it rather than obey.
    if chosen != "FINISH" and chosen not in available:
        if "writer" in available:
            chosen, instruction = "writer", "Write the final answer from what the team has gathered."
            note = f"(overrode {decision.next_worker}: already used up)"
        else:
            return finish(f"{decision.next_worker} is used up and no writer left → FINISH")
    else:
        note = ""

    return {
        "next_worker": chosen,
        "instruction": instruction,
        "steps": step,
        "scratchpad": [f"[supervisor → {chosen}] {instruction} {note}".strip()],
    }


def make_worker_node(name: str, agent):
    """Wrap an agent as a graph node that appends to the shared scratchpad."""

    def node(state: TeamState) -> dict:
        task = (
            f"User's question: {state['question']}\n\n"
            f"Your instruction: {state['instruction']}\n\n"
            f"What the team knows so far:\n" + ("\n\n".join(state["scratchpad"][-4:]))
        )
        output = agent(task)
        update = {"scratchpad": [f"[{name}] {output}"], "used": [name]}
        if name == "writer":
            update["final_answer"] = output
        return update

    return node


def route(state: TeamState) -> str:
    return state["next_worker"]


builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("librarian", make_worker_node("librarian", researcher))
builder.add_node("analyst", make_worker_node("analyst", analyst))
builder.add_node("writer", make_worker_node("writer", writer))

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route,
    {"librarian": "librarian", "analyst": "analyst", "writer": "writer", "FINISH": END},
)
# Every worker reports back to the supervisor. That's the cycle.
for worker in ["librarian", "analyst", "writer"]:
    builder.add_edge(worker, "supervisor")

team = builder.compile()
draw(team)

# %%
def run_team(question: str, graph=team):
    """Run the team and print a readable transcript of who did what."""
    print(f"\n{'=' * 74}\n❓ {question}\n{'=' * 74}")
    final = None
    for update in graph.stream(
        {"question": question, "scratchpad": [], "used": [], "final_answer": "",
         "next_worker": "", "instruction": "", "steps": 0},
        {"recursion_limit": 30},
    ):
        for node_name, output in update.items():
            if node_name == "supervisor":
                print(f"\n🎩 SUPERVISOR → {output['next_worker']}")
                if output.get("instruction"):
                    print(f"   \"{output['instruction']}\"")
            else:
                body = output["scratchpad"][0].split("] ", 1)[-1]
                icon = {"librarian": "📚", "analyst": "🧮", "writer": "✍️ "}[node_name]
                print(f"{icon} {node_name.upper()}: {body.strip()[:420]}")
            if output.get("final_answer"):
                final = output["final_answer"]
    print(f"\n{'─' * 74}\n🏁 FINAL ANSWER\n{'─' * 74}\n{(final or '(none)').strip()}")
    return final


run_team("How much does the institute spend per employee per year, "
         "and how does that compare to a postdoc's annual travel budget?")

# %% [markdown]
# Read the transcript. The supervisor sent the librarian for facts, the analyst
# for the division, then the writer — and it decided that ordering itself.
# Nothing in our code says "search before you calculate".
#
# **What the three guards in `supervisor_node` are doing.** Each one was added
# after watching this exact graph misbehave: first it sent the writer five times
# in a row with an identical instruction; once that was blocked, it sent the
# *librarian* five times, getting the same "not in the handbook" answer each
# time. A 2B supervisor simply cannot keep track of what it has already done.
#
# The lesson generalises far beyond this notebook:
#
# > **Facts go in the state and are checked in code. Judgement goes to the
# > model.** "Has the writer already answered?" is a fact. "Who should work
# > next?" is a judgement.
#
# Mixing the two is the most common reason a supervisor loops. The `used` field
# exists for the same reason — we tell the supervisor who has already worked
# rather than hoping it remembers.

# %%
# Now a question that needs no research at all. A good supervisor skips workers.
run_team("What is 96 multiplied by 4?")

# %%
# And one the knowledge base cannot answer. Watch whether the team admits it
# rather than inventing something -- this is where multi-agent systems most
# often go wrong, because each hop is a chance to launder a guess into a fact.
run_team("What is the institute's policy on working from home?")

# %% [markdown]
# ## Pattern 3 · Network / handoffs — agents call each other directly
#
# No supervisor. Each agent can hand off to any other agent by returning a
# `Command(goto=...)`. Cheaper (no supervisor call per hop) and more flexible,
# but much harder to reason about: any agent can send work anywhere.
#
# Use it when the agents are genuinely peers. Avoid it when you need to explain
# the system's behaviour to someone else.

# %%
from langgraph.types import Command


class NetworkState(TypedDict):
    question: str
    log: Annotated[list[str], operator.add]
    answer: str
    hops: int


class Handoff(BaseModel):
    """Either answer, or pass the work to a colleague."""

    action: Literal["answer", "handoff"]
    to: Literal["facts", "maths", "none"] = Field(
        description="which colleague to hand to; 'none' when answering"
    )
    content: str = Field(description="your answer, or the question for your colleague")


def peer(name: str, persona: str, colleagues: str):
    def node(state: NetworkState) -> Command:
        if state["hops"] >= 4:                     # the mandatory circuit breaker
            return Command(goto=END, update={"answer": "Gave up: too many hops.",
                                             "log": [f"[{name}] hop limit"]})

        decision = llm.with_structured_output(Handoff).invoke(
            f"{persona}\n\nYour colleagues: {colleagues}\n\n"
            f"Question: {state['question']}\n\n"
            f"Conversation so far:\n" + "\n".join(state["log"][-4:]) +
            "\n\nEither answer if you can, or hand off to the right colleague."
        )
        log = [f"[{name}] {decision.action} → {decision.to}: {decision.content[:200]}"]

        if decision.action == "answer":
            return Command(goto=END, update={"answer": decision.content, "log": log})
        return Command(
            goto=decision.to if decision.to != "none" else END,
            update={"log": log, "hops": state["hops"] + 1},
        )

    return node


builder = StateGraph(NetworkState)
builder.add_node(
    "facts",
    peer("facts", "You know the institute handbook by heart. Here it is:\n"
                  + handbook[:3500],
         "'maths' does arithmetic."),
)
builder.add_node(
    "maths",
    peer("maths", "You are excellent at arithmetic but know nothing about the "
                  "institute.",
         "'facts' knows the institute handbook."),
)
builder.add_edge(START, "facts")

network = builder.compile()

# ⚠ Run this cell a few times. You will often watch the two agents hand the
# same half-finished sentence back and forth until the hop limit stops them.
# That is not a bug in the example -- it is the network pattern's signature
# failure mode, and it is exactly why the supervisor pattern is the default.

out = network.invoke(
    {"question": "The institute has 214 staff and 4 departments. "
                 "What is the average department size?",
     "log": [], "answer": "", "hops": 0},
    {"recursion_limit": 20},
)
for line in out["log"]:
    print(" ", line)
print("\n🏁", out["answer"])

# %% [markdown]
# ## Pattern 4 · Hierarchical teams
#
# Supervisors of supervisors. A top-level coordinator routes to *team* leads,
# each of which routes to its own workers.
#
# ```
#                  ┌── coordinator ──┐
#                  ▼                 ▼
#          research lead        writing lead
#           ▼        ▼            ▼      ▼
#       searcher  verifier     drafter  editor
# ```
#
# We won't build one today — it is the supervisor pattern nested, and the code
# is the same code twice. What matters is knowing *when*: only once a single
# supervisor is choosing between more than about seven workers, or when whole
# subsystems are owned by different teams. Below that, nesting buys you latency
# and nothing else.

# %% [markdown]
# ## Choosing a pattern
#
# | Pattern | Who decides the flow | Best for | Main risk |
# |---------|---------------------|----------|-----------|
# | **Pipeline** | you, at design time | fixed, well-understood workflows | wasted steps; no adaptivity |
# | **Supervisor** | one router agent | most things — start here | bottleneck; a hop costs a call |
# | **Network** | every agent | peer specialists, fluid problems | loops, unpredictability |
# | **Hierarchical** | nested supervisors | large systems, many teams | latency, complexity |
# | **Debate / critic** | fixed cycle (nb 08) | quality-critical writing, review | cost per revision |
#
# ## The three rules that actually keep these systems working
#
# 1. **Every cycle needs a hard budget.** Both graphs above have one. This is
#    not defensive programming, it is a requirement; agents loop.
# 2. **Print the trace.** You cannot debug a multi-agent system from its final
#    answer. `stream()` and a readable transcript are the minimum. (For real
#    projects: LangSmith, or any OpenTelemetry tracer.)
# 3. **Make roles narrow and their outputs structured.** "Report only what the
#    handbook says" is a testable contract. "Be helpful" is not.
#
# → Next: **10 · Capstone** — we assemble everything into one research desk,
# with a critic loop, a human approval gate, and full observability.

# %% [markdown]
# ## Your turn — 20 minutes
#
# 1. Add a **Critic** worker to the supervisor team: it reads the writer's
#    output and either approves it or sends it back. (You already built this in
#    notebook 08 — here it is a fourth worker and one line in `WORKERS`.)
# 2. Give the writer a *different* model from the supervisor — e.g. supervisor on
#    `litellm`, workers on `ollama`. Where does quality change most?
# 3. Break it on purpose: delete the "Use LAST" hint from the writer's
#    description. Does the supervisor now call the writer first?
# 4. Ask a question needing two *separate* handbook lookups. Does the supervisor
#    call the librarian twice, or does it give up after one?
# 5. Lower the step budget to 2. What does the system do when it runs out —
#    and is the failure obvious to the user? (It should be.)
