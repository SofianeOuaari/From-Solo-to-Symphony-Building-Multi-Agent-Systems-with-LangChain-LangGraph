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
# # 02 · Your first LLM call with LangChain
#
# **The one idea in this notebook:** a language model is a *function*.
# Text in, text out, plus a little randomness.
#
# Everything else we build today — RAG, tools, agents, multi-agent graphs — is
# scaffolding around that function. So let's get very comfortable with it.
#
# We will go, in order:
#
# 1. call a model with a plain string
# 2. call it with **messages** (system / user / assistant) — the real interface
# 3. see what `temperature` does
# 4. **stream** the answer token by token
# 5. build a reusable **prompt template**
# 6. chain things together with the `|` operator (LCEL)
# 7. swap Ollama ↔ the proxy *without touching the chain*

# %%
# %pip install -q -r ../requirements.txt

# %%
import pathlib
import sys

ROOT = next(
    p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir()
)
sys.path.insert(0, str(ROOT))

from common.workshop_setup import describe_model, get_chat_model, show, timer

# %% [markdown]
# ## 1 · The simplest possible call
#
# `get_chat_model()` uses whatever `WORKSHOP_BACKEND` says in `.env`. Let's see
# what we got, then call it.

# %%
llm = get_chat_model(temperature=0.2, max_tokens=300)
print("we are using:", describe_model(llm))

# %%
with timer("first call"):
    response = llm.invoke("In two sentences: what is a multi-agent LLM system?")

show(response)

# %% [markdown]
# Two things to notice:
#
# * **The first call is slow.** The model has to be loaded into memory (Ollama)
#   or your request has to queue (shared proxy). Later calls are much faster —
#   re-run the cell above and compare.
# * **`invoke` returned an object, not a string.** It's an `AIMessage`.

# %%
print("type      :", type(response).__name__)
print("content   :", repr(response.content[:80]), "...")
print("token usage:", response.usage_metadata)
print("\nmodel's own metadata:")
for key, value in list(response.response_metadata.items())[:6]:
    print(f"  {key}: {value}")

# %% [markdown]
# ## 2 · Messages: the interface that actually exists
#
# Under the hood, a chat model never sees a bare string. It sees a **list of
# messages**, each with a role:
#
# | Role | LangChain class | What it is for |
# |------|-----------------|----------------|
# | system | `SystemMessage` | the standing instructions: who the model is, rules |
# | user | `HumanMessage` | what the person just said |
# | assistant | `AIMessage` | what the model said before (this is how "memory" works!) |
#
# When you pass a plain string, LangChain silently wraps it in one
# `HumanMessage`. Let's do it explicitly — the system message is where an
# agent's **role** lives, and roles are the foundation of everything after
# notebook 07.

# %%
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(
        "You are a terse statistics tutor for PhD students. "
        "Answer in at most 3 sentences. Never apologise."
    ),
    HumanMessage("Why is a p-value not the probability that my hypothesis is false?"),
]

show(llm.invoke(messages))

# %% [markdown]
# ### The same model, a different role
#
# Change *only* the system message and the model behaves like a different tool.
# This is literally how we will give our agents distinct personalities later:
# a "Researcher", a "Critic", a "Writer" can all be the *same* model with three
# different system messages.

# %%
for role in [
    "You are a pirate. Answer in one sentence.",
    "You are a formal peer reviewer for Nature. Answer in one sentence.",
    "You are a 5-year-old. Answer in one sentence.",
]:
    reply = llm.invoke([SystemMessage(role), HumanMessage("What is a p-value?")])
    print(f"» {role}\n  {reply.content.strip()[:200]}\n")

# %% [markdown]
# ### Memory, demystified
#
# There is no memory in a language model. Every call is independent. "Memory"
# is just *you resending the earlier messages*. Watch:

# %%
# Without history -- the model has no idea what "it" refers to.
print("NO HISTORY:")
print(llm.invoke([HumanMessage("And how do I compute it in Python?")]).content[:200])

# %%
# With history -- suddenly it knows.
print("WITH HISTORY:")
conversation = [
    SystemMessage("You are a terse statistics tutor. Max 2 sentences."),
    HumanMessage("Explain the Mann-Whitney U test."),
    AIMessage("It is a non-parametric test comparing whether one of two samples tends to have larger values."),
    HumanMessage("And how do I compute it in Python?"),
]
print(llm.invoke(conversation).content[:300])

# %% [markdown]
# Hold on to this. In notebook 08, LangGraph's job will be precisely to carry
# that growing message list around for us — automatically, and across agents.

# %% [markdown]
# ## 3 · Temperature: the randomness dial
#
# `temperature=0` → always pick the most likely next token (repeatable).
# `temperature=1` → sample more adventurously (creative, less reliable).
#
# For agents that *make decisions* you generally want it low. For brainstorming,
# high.

# %%
prompt = "Invent a name for a research group that studies bird migration. Name only."

for temperature in [0.0, 0.0, 1.2, 1.2]:
    model = get_chat_model(temperature=temperature, max_tokens=40)
    answer = model.invoke(prompt).content.strip().replace("\n", " ")
    print(f"temp={temperature}:  {answer[:80]}")

# %% [markdown]
# The two `temp=0.0` runs should be (near) identical; the two `temp=1.2` runs
# should differ. *Near* identical, not guaranteed identical — GPU/threading
# non-determinism means even temperature 0 is not a promise.

# %% [markdown]
# ## 4 · Streaming
#
# `invoke` waits for the whole answer. `stream` yields chunks as they are
# generated. Nothing about the model changes; it is purely about perceived
# latency — and it's one line of code.

# %%
print("streaming: ", end="", flush=True)
for chunk in llm.stream("List three orchestration patterns for multi-agent systems."):
    print(chunk.content, end="", flush=True)
print()

# %% [markdown]
# ## 5 · Prompt templates
#
# Hard-coding prompts with f-strings works until you have twelve of them. A
# `ChatPromptTemplate` is a *reusable, parameterised* prompt.

# %%
from langchain_core.prompts import ChatPromptTemplate

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {persona}. Answer in at most {n_sentences} sentences."),
        ("human", "Explain {topic} to me."),
    ]
)

# Filling it in gives you... a list of messages. No magic.
filled = summary_prompt.invoke(
    {"persona": "patient methods teacher", "n_sentences": 2, "topic": "cross-validation"}
)
for message in filled.messages:
    print(f"[{message.type}] {message.content}")

# %% [markdown]
# ## 6 · Chains: the `|` operator (LCEL)
#
# LangChain lets you pipe components together, like a shell pipeline:
#
# ```
# dict  ->  prompt  ->  model  ->  parser  ->  str
# ```
#
# Each piece is a "Runnable" and every Runnable has `.invoke()`, `.stream()`,
# `.batch()`. So the *chain* has them too. That composability is the reason the
# library exists.

# %%
from langchain_core.output_parsers import StrOutputParser

chain = summary_prompt | llm | StrOutputParser()

print(
    chain.invoke(
        {"persona": "patient methods teacher", "n_sentences": 2, "topic": "cross-validation"}
    )
)

# %%
# Because it's a Runnable, we get batching for free -- and it runs concurrently.
topics = ["overfitting", "statistical power", "a random effect"]

with timer("3 topics in one batch"):
    answers = chain.batch(
        [{"persona": "methods teacher", "n_sentences": 1, "topic": t} for t in topics]
    )

for topic, answer in zip(topics, answers):
    print(f"\n### {topic}\n{answer.strip()}")

# %%
# And streaming, through the whole chain:
for piece in chain.stream(
    {"persona": "methods teacher", "n_sentences": 3, "topic": "bootstrapping"}
):
    print(piece, end="", flush=True)
print()

# %% [markdown]
# ## 7 · Swapping the backend
#
# Here is the payoff. The chain below is *not rebuilt*. We only replace the
# model object inside it, and the same code now runs against a 35B model on a
# GPU cluster instead of a 2B model on your CPU.

# %%
question = {
    "persona": "rigorous methodologist",
    "n_sentences": 3,
    "topic": "why multiple comparisons corrections matter",
}

for backend in ["ollama", "litellm"]:
    print(f"\n{'=' * 70}\n{backend.upper()}\n{'=' * 70}")
    try:
        model = get_chat_model(backend=backend, temperature=0.2, max_tokens=800)
        print(f"({describe_model(model)})\n")
        with timer(backend):
            # Same prompt, same parser -- only the middle link changed.
            print((summary_prompt | model | StrOutputParser()).invoke(question).strip())
    except Exception as exc:
        print(f"skipped: {type(exc).__name__}: {exc}")

# %% [markdown]
# ### A wrinkle worth knowing: thinking models and empty answers
#
# Almost every model we use today (Qwen3.5, Qwen3.6, DeepSeek-R1) can write a
# long **private chain of thought** before the answer you see. It usually helps
# accuracy — but it is slow, and those thinking tokens are charged against your
# `max_tokens` budget.
#
# Consequence: a thinking model with a small `max_tokens` returns an **empty
# string**. It spent the whole budget thinking and never reached the answer.
#
# Our helper therefore ships with `thinking=False` by default, so that today's
# demos are fast and predictable. Let's switch it on and watch it happen:

# %%
thinker = get_chat_model(thinking=True, max_tokens=24)

reply = thinker.invoke("What is 17 * 23?")
print("content  :", repr(reply.content))
print("thinking :", repr(reply.additional_kwargs.get("reasoning_content", ""))[:250])
print("\n^ empty or truncated: the 24-token budget went to thinking.")

# %%
# Same model, same question, room to breathe:
reply = get_chat_model(thinking=True, max_tokens=3000).invoke("What is 17 * 23?")
show(reply)

# %% [markdown]
# **The debugging rule:** blank reply → raise `max_tokens` (or set
# `thinking=False`) *before* you start rewriting your prompt.
#
# We come back to reasoning properly in notebook 06, where we measure whether
# all that thinking actually buys better answers.

# %% [markdown]
# ## Your turn — 5 minutes
#
# 1. Write a system message that makes the model answer **only** in bullet
#    points, and check that it obeys. Then try to make it disobey.
# 2. Build a `ChatPromptTemplate` that translates text into a language of your
#    choice, chain it, and `.batch()` five sentences through it.
# 3. Set `temperature=2.0`. What happens? Why is that a bad idea for an agent
#    that has to choose which tool to call?
#
# → Next: **03 · Embeddings and cosine similarity** — the same model family,
# but instead of text we take out *numbers*, and suddenly we can measure
# meaning.
