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
# # 08 · LangGraph fundamentals
#
# In notebook 07 we wrote the agent loop by hand, then let `create_agent` hide
# it again. That is fine for *one* agent with a flat list of tools. It is not
# fine when you want:
#
# * a **critic** that can send work back for revision (a cycle)
# * a **router** that sends different questions down different paths
# * a **human** approving a step before it runs
# * **several agents** handing work to each other
#
# For that you need to describe the control flow explicitly. That is LangGraph.
#
# > **The mental model.**
# > **State** — one shared dictionary that flows through the system.
# > **Nodes** — functions that read the state and return an update to it.
# > **Edges** — which node runs next. Conditional edges let the *data* decide.
#
# It is a state machine, and everything from here on is built out of these three
# pieces.

# %%
# %pip install -q -r ../requirements.txt

# %%
import pathlib
import sys

ROOT = next(
    p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir()
)
sys.path.insert(0, str(ROOT))

from common.workshop_setup import describe_model, draw, get_chat_model

llm = get_chat_model(temperature=0, max_tokens=600)
print("using:", describe_model(llm))

# %% [markdown]
# ## 1 · A graph with no LLM in it at all
#
# Start with something you can fully predict. Three nodes in a line.

# %%
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    """The shared state. Every node sees this; every node may update it."""

    value: int
    log: list[str]


def double(state: CounterState) -> dict:
    # A node RETURNS AN UPDATE -- a dict with the keys it wants to change.
    # It does not mutate the state in place.
    new_value = state["value"] * 2
    return {"value": new_value, "log": state["log"] + [f"doubled → {new_value}"]}


def add_ten(state: CounterState) -> dict:
    new_value = state["value"] + 10
    return {"value": new_value, "log": state["log"] + [f"added 10 → {new_value}"]}


builder = StateGraph(CounterState)
builder.add_node("double", double)
builder.add_node("add_ten", add_ten)

builder.add_edge(START, "double")     # START is where execution enters
builder.add_edge("double", "add_ten")
builder.add_edge("add_ten", END)      # END is where it stops

graph = builder.compile()

result = graph.invoke({"value": 5, "log": []})
print(result["value"])
for line in result["log"]:
    print("  ", line)

# %% [markdown]
# ### Look at what you built
#
# LangGraph can draw the graph. Do this constantly — a picture of your control
# flow is the main thing you get from using a graph framework at all.

# %%
# `draw()` is a tiny helper in common/workshop_setup.py: it renders a PNG in a
# notebook, falls back to ASCII art, and finally to raw Mermaid source.
draw(graph)

# %%
# The ASCII version works everywhere and is what you want in a terminal:
print(graph.get_graph().draw_ascii())

# %% [markdown]
# ## 2 · Conditional edges — letting the data choose
#
# A straight line is just a function. The moment a graph earns its keep is when
# the *next step depends on the state*.
#
# A conditional edge is a function that looks at the state and returns the
# **name of the next node**.

# %%
class TriageState(TypedDict):
    number: int
    trail: list[str]


def classify(state: TriageState) -> dict:
    return {"trail": state["trail"] + ["classified"]}


def handle_small(state: TriageState) -> dict:
    return {"trail": state["trail"] + [f"{state['number']} is small"]}


def handle_large(state: TriageState) -> dict:
    return {"trail": state["trail"] + [f"{state['number']} is LARGE"]}


def choose_branch(state: TriageState) -> str:
    """A router. Returns the NAME of the next node."""
    return "large" if state["number"] > 100 else "small"


builder = StateGraph(TriageState)
builder.add_node("classify", classify)
builder.add_node("small", handle_small)
builder.add_node("large", handle_large)

builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    choose_branch,
    {"small": "small", "large": "large"},   # map return value → node name
)
builder.add_edge("small", END)
builder.add_edge("large", END)

triage = builder.compile()
draw(triage)

for number in [7, 5000]:
    print(number, "→", triage.invoke({"number": number, "trail": []})["trail"])

# %% [markdown]
# ## 3 · Reducers — how state gets *combined*
#
# You may have noticed the clumsy `state["log"] + [...]` above. Every node had
# to remember to append rather than overwrite. With many nodes that is a bug
# factory.
#
# A **reducer** moves that rule into the state definition: "when a node returns
# a value for this key, *add* it to what's there instead of replacing it."

# %%
import operator
from typing import Annotated


class ReducerState(TypedDict):
    # No reducer: last writer wins.
    current: str
    # With a reducer: every node's contribution is concatenated.
    history: Annotated[list[str], operator.add]


def node_a(state: ReducerState) -> dict:
    return {"current": "A", "history": ["A ran"]}      # just the new bit!


def node_b(state: ReducerState) -> dict:
    return {"current": "B", "history": ["B ran"]}


builder = StateGraph(ReducerState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

out = builder.compile().invoke({"current": "", "history": []})
print("current (overwritten):", out["current"])
print("history (accumulated):", out["history"])

# %% [markdown]
# ### The reducer you will use every single time
#
# For conversations, LangGraph ships `add_messages`. It appends new messages,
# and — crucially — *replaces* a message if it has the same id, which is how
# streaming and edits work.
#
# `MessagesState` is a ready-made TypedDict containing exactly that one field.
# Almost every agent graph you write starts from it.

# %%
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

print("MessagesState is just:", MessagesState.__annotations__)

# %% [markdown]
# ## 4 · A real graph: classify → route → respond
#
# Now with LLMs in the nodes. This is the shape of a customer-support triage
# system, and it is the direct ancestor of the supervisor in notebook 09.

# %%
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class Category(BaseModel):
    """Route a researcher's request to the right desk.

    technical      = computers, clusters, software, storage, accounts, data loss
    administrative = money, forms, deadlines, contracts, travel, reimbursement
    scientific     = study design, statistics, choice of method, interpretation
    """

    # ⚠ FIELD ORDER MATTERS. A model fills a JSON object left to right, so a
    # field declared *before* the decision becomes its scratch space -- and one
    # declared after is just a post-hoc excuse for a choice already made.
    # Putting `reason` first is chain-of-thought smuggled into the schema, and
    # it costs nothing.
    reason: str = Field(
        description="First, in max 15 words, say which desk this belongs to and why."
    )
    kind: Literal["technical", "administrative", "scientific"]


class SupportState(TypedDict):
    question: str
    category: str
    reason: str
    answer: str
    trace: Annotated[list[str], operator.add]


def triage_node(state: SupportState) -> dict:
    """Node 1: decide what kind of question this is."""
    classifier = llm.with_structured_output(Category)
    result = classifier.invoke(
        "Route this request to exactly one desk.\n\n"
        "  technical      — computers, clusters, software, storage, accounts\n"
        "  administrative — money, forms, deadlines, contracts, travel\n"
        "  scientific     — study design, statistics, methods, interpretation\n\n"
        "Examples:\n"
        "  'My disk quota is full'                  → technical\n"
        "  'When is the grant report due?'          → administrative\n"
        "  'Is my sample size large enough?'        → scientific\n\n"
        f"REQUEST: {state['question']}"
    )
    return {
        "category": result.kind,
        "reason": result.reason,
        "trace": [f"triage → {result.kind} ({result.reason})"],
    }


def make_specialist(name: str, persona: str):
    """Build a node that answers in a particular voice. Same model, new role."""

    def specialist(state: SupportState) -> dict:
        answer = llm.invoke(
            [SystemMessage(persona), HumanMessage(state["question"])]
        ).content
        return {"answer": answer, "trace": [f"answered by {name}"]}

    return specialist


def route_by_category(state: SupportState) -> str:
    return state["category"]


builder = StateGraph(SupportState)
builder.add_node("triage", triage_node)
builder.add_node(
    "technical",
    make_specialist("IT support",
                    "You are the IAS IT helpdesk. Be practical and concrete. "
                    "Max 3 sentences, and end with the next concrete step."),
)
builder.add_node(
    "administrative",
    make_specialist("Admin office",
                    "You are the IAS administrative office. Be formal, mention "
                    "forms and deadlines. Max 3 sentences."),
)
builder.add_node(
    "scientific",
    make_specialist("Methods consultant",
                    "You are a statistics consultant. Be precise, name specific "
                    "methods. Max 3 sentences."),
)

builder.add_edge(START, "triage")
builder.add_conditional_edges(
    "triage",
    route_by_category,
    {"technical": "technical", "administrative": "administrative",
     "scientific": "scientific"},
)
for node in ["technical", "administrative", "scientific"]:
    builder.add_edge(node, END)

support = builder.compile()
draw(support)

# %%
QUESTIONS = [
    "My SLURM job keeps dying after 24 hours with no error message.",
    "How do I claim reimbursement for a conference I attended in May?",
    "Should I use a mixed model or a repeated-measures ANOVA for my nested design?",
]

for question in QUESTIONS:
    result = support.invoke(
        {"question": question, "category": "", "reason": "", "answer": "", "trace": []}
    )
    print(f"\nQ: {question}")
    for line in result["trace"]:
        print(f"   · {line}")
    print(f"   → {result['answer'].strip()[:260]}")

# %% [markdown]
# ### Two things fixed a router that classified everything as "technical"
#
# The first draft of this notebook had `kind` declared before `reason`, and a
# bare `Literal` with no explanation. It sent *every* question to the same desk —
# and the `reason` field would cheerfully say *"this falls under the
# administrative category"* while `kind` said `technical`.
#
# 1. **Field order.** A model writes a JSON object left to right. `reason`
#    declared *first* is scratch space it can think in; declared *last* it is a
#    rationalisation of a choice already made. Free chain-of-thought.
# 2. **Say what the labels mean.** The docstring, the descriptions and three
#    one-line examples do more than any amount of "be accurate".
#
# > **Your schema is your prompt.** Try swapping the two fields back and
# > re-running — it is the cheapest experiment in the workshop.
#
# Even so, a 2B model still gets roughly two of these three right: the
# mixed-model question usually lands on `technical`, because "model" and
# "design" read as engineering words to a small model. Run the same cell on
# `litellm` and it goes to `scientific`.
#
# **The general point:** a router's accuracy caps the accuracy of everything
# downstream of it. If you build a supervisor (notebook 09), measure the router
# on its own, with a list of questions and known answers, before you blame the
# specialists.

# %% [markdown]
# ### Watching it run, step by step
#
# `.stream()` yields the state update after every node. This is your debugger:
# when a graph misbehaves, stream it and watch where the state goes wrong.

# %%
for update in support.stream(
    {"question": "The /scratch directory ate my data.", "category": "",
     "reason": "", "answer": "", "trace": []}
):
    for node_name, node_output in update.items():
        keys = {k: str(v)[:60] for k, v in node_output.items()}
        print(f"[{node_name}] {keys}")

# %% [markdown]
# ## 5 · Memory: making the graph remember
#
# Everything so far was stateless: each `invoke` started from nothing. Add a
# **checkpointer** and LangGraph saves the state after every node, keyed by a
# `thread_id`. Two lines of code and you have a multi-turn conversation.
#
# (`InMemorySaver` for the workshop; `SqliteSaver` or `PostgresSaver` when you
# want it to survive a restart.)

# %%
from langgraph.checkpoint.memory import InMemorySaver


def chat_node(state: MessagesState) -> dict:
    system = SystemMessage(
        "You are a concise assistant at a research institute. Max 2 sentences."
    )
    # With add_messages, returning one message APPENDS it.
    return {"messages": [llm.invoke([system] + state["messages"])]}


chat_builder = StateGraph(MessagesState)
chat_builder.add_node("chat", chat_node)
chat_builder.add_edge(START, "chat")
chat_builder.add_edge("chat", END)

chatbot = chat_builder.compile(checkpointer=InMemorySaver())

# %%
# A thread_id names the conversation. Same id = same memory.
alice = {"configurable": {"thread_id": "alice"}}

for turn in [
    "I work in Department C on lake sensors.",
    "What department did I say I work in?",
    "Suggest one analysis method suitable for my data.",
]:
    reply = chatbot.invoke({"messages": [HumanMessage(turn)]}, config=alice)
    print(f"\n👤 {turn}\n🤖 {reply['messages'][-1].content.strip()}")

# %%
# A different thread_id is a different person, with no shared memory.
bob = {"configurable": {"thread_id": "bob"}}
reply = chatbot.invoke(
    {"messages": [HumanMessage("What department did I say I work in?")]}, config=bob
)
print("🤖 (bob's thread):", reply["messages"][-1].content.strip())

# %%
# The whole history is inspectable -- and rewindable.
state = chatbot.get_state(alice)
print(f"alice's thread has {len(state.values['messages'])} messages:\n")
for message in state.values["messages"]:
    who = "👤" if isinstance(message, HumanMessage) else "🤖"
    print(f"  {who} {message.content.strip()[:80]}")

# %% [markdown]
# ## 6 · Cycles — a graph that revises its own work
#
# Here is what you genuinely cannot do with a chain: **go backwards**. A writer
# node drafts, a critic node judges, and if the critic is unhappy the graph
# loops back to the writer.
#
# This draft→critique→revise cycle is the single most useful multi-agent pattern
# there is, and in notebook 10 it becomes the heart of our final system.

# %%
class DraftState(TypedDict):
    task: str
    draft: str
    critique: str
    approved: bool
    revisions: int


class Review(BaseModel):
    """A critic's verdict on a draft."""

    approved: bool = Field(description="true only if the draft fully meets the brief")
    critique: str = Field(description="what to fix, max 25 words; empty if approved")


def write(state: DraftState) -> dict:
    if state["revisions"] == 0:
        prompt = f"Write this, 2 sentences maximum:\n\n{state['task']}"
    else:
        prompt = (
            f"Task: {state['task']}\n\n"
            f"Your previous draft:\n{state['draft']}\n\n"
            f"An editor said: {state['critique']}\n\n"
            f"Rewrite it, 2 sentences maximum. Address the criticism."
        )
    draft = llm.invoke(prompt).content
    return {"draft": draft, "revisions": state["revisions"] + 1}


def critique(state: DraftState) -> dict:
    critic = llm.with_structured_output(Review)
    review = critic.invoke(
        "You are a strict editor. Judge whether this draft meets the brief.\n\n"
        f"BRIEF: {state['task']}\n\nDRAFT: {state['draft']}"
    )
    return {"approved": review.approved, "critique": review.critique}


def should_continue(state: DraftState) -> str:
    """The exit condition. NEVER build a cycle without one."""
    if state["approved"]:
        return "done"
    if state["revisions"] >= 3:          # hard cap beats infinite loop
        return "done"
    return "revise"


builder = StateGraph(DraftState)
builder.add_node("write", write)
builder.add_node("critique", critique)
builder.add_edge(START, "write")
builder.add_edge("write", "critique")
builder.add_conditional_edges(
    "critique", should_continue, {"revise": "write", "done": END}
)

writer_critic = builder.compile()
draw(writer_critic)

# %%
TASK = ("A tweet announcing that the PhD programme at the Institute for Adaptive "
        "Systems (a research institute in Konstanz) is open for applications. "
        "It must mention the 4-year duration and include a call to action.")

for update in writer_critic.stream(
    {"task": TASK, "draft": "", "critique": "", "approved": False, "revisions": 0}
):
    for node_name, output in update.items():
        if node_name == "write":
            print(f"\n✍️  DRAFT {output['revisions']}: {output['draft'].strip()[:220]}")
        else:
            verdict = "✅ approved" if output["approved"] else "❌ " + output["critique"]
            print(f"🧐 CRITIC: {verdict}")

# %% [markdown]
# **That loop is the whole idea of multi-agent collaboration**, with two agents
# and one shared state. Notebook 09 adds specialised roles and a supervisor;
# notebook 10 adds tools and a human. But the machinery is what you just built.

# %% [markdown]
# ## 7 · Human-in-the-loop
#
# For anything consequential — sending an email, spending money, writing to a
# database — you want a person to approve first. `interrupt()` pauses the graph
# mid-node and hands control back to you; resuming continues from exactly there.
#
# It works because the checkpointer already saves state after every node. The
# pause is just a checkpoint you don't automatically resume from.

# %%
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    request: str
    draft: str
    sent: bool


def draft_email(state: ApprovalState) -> dict:
    email = llm.invoke(
        f"Write a 2-sentence professional email for: {state['request']}"
    ).content
    return {"draft": email}


def send_email(state: ApprovalState) -> dict:
    # Execution STOPS here and returns control to the caller.
    decision = interrupt({"draft": state["draft"], "question": "Send this email?"})
    if decision == "approve":
        print("   📧 (pretending to send)")
        return {"sent": True}
    return {"sent": False}


builder = StateGraph(ApprovalState)
builder.add_node("draft", draft_email)
builder.add_node("send", send_email)
builder.add_edge(START, "draft")
builder.add_edge("draft", "send")
builder.add_edge("send", END)

approver = builder.compile(checkpointer=InMemorySaver())

# %%
thread = {"configurable": {"thread_id": "email-1"}}

paused = approver.invoke(
    {"request": "decline a peer review invitation politely", "draft": "", "sent": False},
    config=thread,
)

print("⏸️  graph paused, waiting for a human.\n")
print("DRAFT:\n", paused["__interrupt__"][0].value["draft"].strip())
print("\nQUESTION:", paused["__interrupt__"][0].value["question"])

# %%
# A human decides. Resume by passing a Command back in.
final = approver.invoke(Command(resume="approve"), config=thread)
print("sent:", final["sent"])

# %%
# The same graph, on a different thread, rejected instead:
thread2 = {"configurable": {"thread_id": "email-2"}}
approver.invoke(
    {"request": "ask for a deadline extension", "draft": "", "sent": False}, config=thread2
)
print("sent:", approver.invoke(Command(resume="reject"), config=thread2)["sent"])

# %% [markdown]
# ## What you can now build
#
# | Concept | What it gives you |
# |---------|-------------------|
# | `StateGraph` + `TypedDict` | one shared state, explicit |
# | nodes | steps that read state and return updates |
# | conditional edges | the data chooses the path |
# | reducers (`add_messages`) | state that accumulates safely |
# | checkpointer + `thread_id` | memory across turns, and rewind |
# | cycles + exit condition | self-correction |
# | `interrupt()` | a human in the decision |
#
# Every one of these appears in the next two notebooks. The only thing left to
# add is **more than one role** — which turns out to be surprisingly little new
# code and a great deal of new design thinking.
#
# → Next: **09 · Multi-agent collaboration**

# %% [markdown]
# ## Your turn — 15 minutes
#
# 1. Add a fourth category (`"ethics"`) to the support graph, with its own
#    specialist node. What do you have to change? (Answer: the `Literal`, the
#    node, the edge map. Three places — that is the cost of a hard-coded router,
#    and the reason notebook 09 does it differently.)
# 2. Make the critic stricter ("reject anything over 20 words") and watch the
#    revision count climb. Then remove the `revisions >= 3` cap and see what
#    happens. Put it back.
# 3. Give `chatbot` a summarisation node that compresses the history once it
#    passes 10 messages. This is how real systems handle long conversations.
# 4. In the approval graph, let the human return *edits* instead of
#    approve/reject — `Command(resume="make it warmer")` — and loop back to the
#    drafting node.
