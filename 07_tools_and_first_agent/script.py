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
# # 07 · Tools, and your first agent
#
# Up to now the model could only *talk*. It cannot check today's date, multiply
# two numbers reliably, read a file, or search your handbook. It can only
# produce plausible text about those things.
#
# **Tools** change that. And once a model can call tools in a loop, deciding for
# itself which one and when, you have an **agent**.
#
# > **A definition worth memorising.**
# > A *chain* is a fixed sequence of steps that you wrote.
# > An **agent** is a loop where *the model* decides the next step.
#
# That is the whole difference. Everything else — memory, roles, multi-agent
# orchestration — is built on this loop.
#
# We will:
#
# 1. write tools and see what the model actually sees
# 2. watch a model *request* a tool call (it cannot run anything itself!)
# 3. **build the agent loop by hand** — 20 lines, no framework
# 4. replace it with LangChain's `create_agent`
# 5. give the agent our RAG knowledge base as a tool
# 6. look at the ways agents fail

# %%
# %pip install -q -r ../requirements.txt

# %%
import pathlib
import sys

ROOT = next(
    p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir()
)
sys.path.insert(0, str(ROOT))

from common.workshop_setup import describe_model, get_chat_model

llm = get_chat_model(temperature=0, max_tokens=800)
print("using:", describe_model(llm))

# %% [markdown]
# ## 1 · A tool is a Python function with a good docstring
#
# The `@tool` decorator turns a function into something a model can call. Note
# what gets sent to the model: **the name, the docstring, and the type hints**.
# Nothing else. The body stays on your machine.
#
# So the docstring is not documentation — **it is the prompt**. A vague docstring
# is the single most common reason an agent "doesn't use my tool".

# %%
from datetime import date

from langchain_core.tools import tool


@tool
def today() -> str:
    """Return today's date in ISO format (YYYY-MM-DD). Use this whenever the
    user asks about the current date, or about how many days until something."""
    return date.today().isoformat()


@tool
def calculate(expression: str) -> str:
    """Evaluate a arithmetic expression and return the result.

    Use this for ANY arithmetic -- do not calculate in your head, you are bad
    at it. The expression must be valid Python using only numbers and the
    operators + - * / ** % ( ), for example "18400000 / (214 * 4)".
    """
    allowed = set("0123456789.+-*/%()e ")
    if not set(expression) <= allowed:
        return f"Error: expression contains forbidden characters: {expression!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Error evaluating {expression!r}: {exc}"


@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


tools = [today, calculate, word_count]

# %%
# This is *exactly* what the model receives about your tool:
import json

for t in tools:
    print(f"name        : {t.name}")
    print(f"description : {t.description}")
    print(f"arguments   : {json.dumps(t.args, indent=2)}")
    print("-" * 70)

# %% [markdown]
# > **The `eval` in `calculate` is a workshop convenience, not a pattern.**
# > We whitelist the characters, which blocks the obvious attacks, but a tool
# > that executes model-generated strings is a serious decision. In production
# > use a real expression parser, and assume that anything a model can reach,
# > it will eventually reach in the worst possible way.

# %% [markdown]
# ## 2 · The model *requests*, it does not *execute*
#
# This is the point everyone gets wrong at first. `bind_tools` tells the model
# what is available. When it wants one, it returns an `AIMessage` with an empty
# `content` and a filled `tool_calls` list. **Nothing has run yet.** It is a
# request, addressed to you.

# %%
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke("What is 18400000 divided by 214, then divided by 4?")

print("content   :", repr(response.content))
print("tool_calls:")
for call in response.tool_calls:
    print(f"   {call['name']}({call['args']})   id={call['id']}")

# %%
# Nothing happened until *we* run it. The model chose the tool and the
# arguments; executing them is our responsibility -- and our safety boundary.
if response.tool_calls:
    call = response.tool_calls[0]
    chosen_tool = {t.name: t for t in tools}[call["name"]]
    result = chosen_tool.invoke(call["args"])
    print("we executed it and got:", result)

# %%
# The model also decides *not* to use a tool when none fits:
plain = llm_with_tools.invoke("What is the capital of Portugal?")
print("tool_calls:", plain.tool_calls)
print("content   :", plain.content[:150])

# %% [markdown]
# ## 3 · The agent loop, by hand
#
# Now the part worth understanding properly. An agent is this loop:
#
# ```
# 1. send the conversation to the model
# 2. did it ask for tools?
#      no  -> we're done, return its answer
#      yes -> run each tool, append the results to the conversation, goto 1
# ```
#
# That's it. Every agent framework in existence — LangGraph, CrewAI, AutoGen,
# OpenAI's Assistants — is this loop plus conveniences. Write it once by hand
# and you will never find agents mysterious again.

# %%
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

TOOLS_BY_NAME = {t.name: t for t in tools}


def run_agent(question: str, max_steps: int = 6, verbose: bool = True) -> str:
    """The ReAct loop, with nothing hidden."""
    messages = [
        SystemMessage(
            "You are a careful research assistant. "
            "Use the provided tools for any date or arithmetic -- never guess. "
            "When you have the answer, state it plainly."
        ),
        HumanMessage(question),
    ]

    for step in range(1, max_steps + 1):
        ai_message = llm_with_tools.invoke(messages)   # 1. ask the model
        messages.append(ai_message)

        if not ai_message.tool_calls:                  # 2. no tools -> done
            if verbose:
                print(f"\n[step {step}] final answer")
            return ai_message.content

        for call in ai_message.tool_calls:             # 3. run what it asked for
            if verbose:
                print(f"[step {step}] → {call['name']}({call['args']})")

            selected = TOOLS_BY_NAME.get(call["name"])
            if selected is None:
                observation = f"Error: no tool named {call['name']}"
            else:
                try:
                    observation = str(selected.invoke(call["args"]))
                except Exception as exc:
                    # Hand the error back to the model -- it can often recover.
                    observation = f"Error: {type(exc).__name__}: {exc}"

            if verbose:
                print(f"           ← {observation[:100]}")

            # The result goes back as a ToolMessage, linked by tool_call_id.
            messages.append(
                ToolMessage(content=observation, tool_call_id=call["id"])
            )

    return "Stopped: hit the step limit without reaching an answer."


# %%
print(run_agent("How many euros per person per year is a budget of 18.4 million "
                "euros over 4 years for 214 people?"))

# %%
# A question needing two different tools, in sequence:
print(run_agent("What is today's date, and how many days are left in this year? "
                "Use the tools."))

# %% [markdown]
# ### Look closely at that first answer
#
# On a small model you will often see something uncomfortable: the tool returns
# `21495.327…`, and the model's sentence then says something *different* — €86,
# or €21.5 thousand, or it silently divides by 4 again.
#
# **The tool was right. The summary was wrong.** Calling a tool guarantees the
# computation is correct; it guarantees nothing about how the model reports it.
# That gap is why notebook 10 puts a *critic* between the work and the user, and
# why "the agent used a calculator" is not the same claim as "the answer is
# right".
#
# If you see it here, you have just found the most important failure mode in the
# workshop. Larger models do it much less — try the same cell on `litellm`.

# %% [markdown]
# Read the trace. Nobody told the agent to call `today` before `calculate` —
# it worked out that it needed the date first. **That ordering decision is the
# agency.**
#
# Note also the `max_steps` guard. An agent without one can loop forever, and
# will. Every framework has this; it is not optional.

# %% [markdown]
# ## 4 · The same thing, from the library
#
# You now know what is inside. Use the built version — it adds streaming, state
# persistence, error handling and the hooks we need in notebook 08.

# %%
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a careful research assistant. "
        "Use the tools for any date or arithmetic -- never guess."
    ),
)

result = agent.invoke(
    {"messages": [HumanMessage("What is 17 * 23 * 41? And how many words are in "
                               "the sentence you just read?")]}
)

# The result is the full conversation -- every step is inspectable.
for message in result["messages"]:
    kind = type(message).__name__
    if getattr(message, "tool_calls", None):
        for call in message.tool_calls:
            print(f"[{kind:<12}] CALL {call['name']}({call['args']})")
    elif message.content:
        print(f"[{kind:<12}] {str(message.content)[:160]}")

# %% [markdown]
# > **This is already a LangGraph graph.** `create_agent` builds a two-node
# > cycle — *model* ⇄ *tools* — with a conditional edge that exits when no tool
# > is requested. In notebook 08 we draw that graph and then build our own.

# %% [markdown]
# ## 5 · Giving the agent a knowledge base
#
# Retrieval was a fixed chain in notebook 05: always search, always answer. As a
# **tool** it becomes a choice. The agent decides whether to search at all, what
# to search for, and whether one search was enough.
#
# That is a genuine upgrade, and it is the bridge to multi-agent systems: a
# "researcher" agent is just an agent whose tool is a retriever.

# %%
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.workshop_setup import get_embeddings

handbook = (ROOT / "05_rag_basics" / "knowledge_base.md").read_text()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=100, separators=["\n## ", "\n\n", "\n", ". ", " "]
).split_text(handbook)

vector_store = InMemoryVectorStore(get_embeddings("nomic-embed-text:latest"))
vector_store.add_texts(chunks)
print(f"indexed {len(chunks)} chunks")


@tool
def search_handbook(query: str) -> str:
    """Search the internal handbook of the Institute for Adaptive Systems (IAS).

    Covers: departments and who leads them, the HELIOS compute cluster, data
    management and DPA-2024, the publication and open-access policy, travel
    budgets, the PhD programme, and ethics approval.

    Pass a full question or a descriptive phrase, not a single keyword.
    Returns the three most relevant passages.
    """
    hits = vector_store.similarity_search(query, k=3)
    return "\n\n---\n\n".join(h.page_content for h in hits)


# %%
research_agent = create_agent(
    model=llm,
    tools=[search_handbook, calculate, today],
    system_prompt=(
        "You are the IAS institute assistant.\n"
        "- For anything about the institute, ALWAYS search the handbook first.\n"
        "- Base your answer only on what the handbook returns.\n"
        "- If the handbook does not cover it, say so plainly.\n"
        "- Use the calculator for arithmetic."
    ),
)


def ask(question: str):
    print(f"\n{'=' * 72}\nQ: {question}\n{'=' * 72}")
    result = research_agent.invoke({"messages": [HumanMessage(question)]})
    for message in result["messages"][1:]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                args = str(call["args"])[:80]
                print(f"  🔧 {call['name']}({args})")
        elif type(message).__name__ == "ToolMessage":
            print(f"  📄 {str(message.content)[:90].strip()}…")
        elif message.content:
            print(f"\n  💬 {message.content.strip()}")
    return result


ask("How many GPUs are there in total on HELIOS, across all nodes?")

# %% [markdown]
# Watch what happened: the agent **searched**, found "96 nodes, 4 GPUs each",
# then **called the calculator** to multiply them. Two tools, chosen and ordered
# by the model, to answer a question that neither tool could answer alone.
#
# In notebook 05 this was impossible — the chain could only paste and answer.

# %%
ask("I'm a PhD student going to a conference in Lisbon. What's my budget, "
    "and what do I need to do before I book?")

# %%
# And the one it should refuse:
ask("What is the institute's policy on office plants?")

# %% [markdown]
# ## 6 · How agents fail
#
# Your participants will hit all four of these today. Naming them in advance
# saves a lot of confusion.

# %%
# Failure 1: the tool exists but the model never calls it, because the
# docstring didn't tell it when to.
@tool
def lookup(x: str) -> str:
    """Looks things up."""          # ← useless description
    return f"the answer for {x} is 42"


vague_agent = create_agent(model=llm, tools=[lookup], system_prompt="Be helpful.")
out = vague_agent.invoke({"messages": [HumanMessage("What is the flumox constant?")]})
used = [c["name"] for m in out["messages"] for c in getattr(m, "tool_calls", []) or []]
print("tools used:", used or "NONE — the docstring never said when to use it")

# %%
# Failure 2: a tool that errors. A good agent reads the error message that comes
# back as a ToolMessage and recovers, rather than crashing.
print(run_agent("Use the calculator to work out 100 divided by 0, "
                "then tell me what happened.", max_steps=4))

# %% [markdown]
# **Failure 3: the loop that never ends.** Two tools whose outputs each suggest
# calling the other, and the agent ping-pongs until `max_steps`. Always set a
# step limit, and log every step so you can see the cycle.
#
# **Failure 4: too many tools.** Past roughly 10–15 tools, selection accuracy
# drops noticeably — the descriptions start to blur together.
#
# That last one is not a bug to fix. It is *the argument for multi-agent
# systems*: instead of one agent with 40 tools, build five agents with 8 tools
# each and a supervisor that routes between them. Each agent then sees a small,
# clean, unambiguous menu.
#
# → Next: **08 · LangGraph fundamentals** — we stop letting the framework hide
# the loop and start drawing it ourselves.

# %% [markdown]
# ## Your turn — 15 minutes
#
# 1. Write a tool that does something useful in your work (read a CSV, query an
#    API, convert units). Add it to `research_agent` and get the agent to use it.
# 2. Take `search_handbook` and **shorten its docstring to three words**. Ask an
#    institute question. Does the agent still search? Now make it too broad
#    ("use this for everything") — does it search when it shouldn't?
# 3. Ask something requiring *three* tools in sequence. Where does it break
#    down? (Small models typically manage 2–3 steps reliably.)
# 4. Set `temperature=1.0` on the agent's model and re-run a question five
#    times. How consistent is the tool selection? What does that tell you about
#    what temperature an agent should run at?
