# verdikt

[![CI](https://github.com/jatin-mishra/verdikt/actions/workflows/ci.yml/badge.svg)](https://github.com/jatin-mishra/verdikt/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/verdikt.svg)](https://pypi.org/project/verdikt/)
[![Python versions](https://img.shields.io/pypi/pyversions/verdikt.svg)](https://pypi.org/project/verdikt/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Configurable LLM-as-judge library for Python.** Plug it into any agent or
pipeline: configure judges in code or YAML, feed in the AI's input/output,
get back a structured, typed `Verdict` — score, label, reasoning, cost, and
token accounting, every time.

verdikt is built as a **library, not a framework**: it has no server, no
required event loop of its own, and no assumptions about your agent stack.
Every extension point — judge types, LLM backends, provider protocols, prompt
templates — is a small, explicit interface you implement and register, so
your own components plug in without forking or patching the library. See
[Extending verdikt](#extending-verdikt) below.

## Contents

- [Features](#features)
- [How it compares](#how-it-compares)
- [Install](#install)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [YAML configuration](#yaml-configuration)
- [Cookbook — judge types](#cookbook--every-judge-type)
- [Cookbook — execution modes](#cookbook--every-execution-mode)
- [Integration cookbook](#integration-cookbook--plugging-verdikt-into-agents)
- [Extending verdikt](#extending-verdikt)
- [Architecture at a glance](#architecture-at-a-glance)
- [Debugging LLM calls](#debugging-llm-calls)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

## Features

- **9 built-in judge types** — `pointwise`, `reference`, `pairwise` (with
  position-swap), `rubric` (G-Eval style), `classifier`,
  `rag_faithfulness`, `rag_context_relevance`, `rag_answer_relevance`,
  `trajectory` (agent runs) — plus your own custom types.
- **Execution modes per judge** — `single` model, `broadcast` to many models
  with consensus (`majority_vote`, `average`, `weighted_average`,
  `unanimous`, `consensus_leader`), or `judge_of_judges` (a meta-judge weighs
  every verdict and its reasoning, not just the score).
- **Multi-step pipelines** — sequential/parallel steps, early-exit gates
  (`on_fail: stop`), conditional steps (`run_if`), and automatic
  `prior_verdicts` passing so later judges see earlier verdicts.
- **Reliability built in** — reasoning-before-score prompts, JSON-only
  outputs, multi-sample voting, retries with backoff, model fallbacks,
  optional response caching, injection-resistant prompt delimiting, and
  cost/token tracking on every verdict.
- **Official provider SDKs** — every call goes through the provider's own
  Python SDK (`anthropic`, `google-genai`); each adapter owns its SDK's
  request/response types entirely, so nothing above it ever has to know.
- **Exact request/response logging** — see the system prompt, every message,
  and the exact reply, tokens, cost and latency for every call, gated behind
  one flag. See [Debugging LLM calls](#debugging-llm-calls).

## How it compares

There's no shortage of LLM-evaluation tooling; verdikt's niche is narrow and
specific: a small, embeddable Python library — not a service, not a
framework you build your whole app around — whose distinct feature is
**panel-style judging**: broadcast to several models, combine them with a
real consensus strategy (`average`, `weighted_average`, `majority_vote`,
`unanimous`, `consensus_leader`), or hand the disagreement to a meta-judge
(`judge_of_judges`) that reads *why* each model scored the way it did, not
just the number. That, plus gated multi-step pipelines, is what the other
projects below generally don't do.

|  | **verdikt** | [DeepEval](https://github.com/confident-ai/deepeval) | [Ragas](https://github.com/explodinggradients/ragas) | [Langfuse](https://github.com/langfuse/langfuse) | [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) | [OpenEvals](https://github.com/langchain-ai/openevals) |
|---|---|---|---|---|---|---|
| What it is | Embeddable Python library | Pytest-style Python/TS eval framework | Python RAG-evaluation framework | Self-hosted/cloud LLM engineering platform | Python framework for model evals (AI-safety oriented, UK AISI) | Small library of prebuilt LangChain evaluator functions |
| Runs as | Import & call — no server | Import & call (+ optional hosted dashboard) | Import & call | A running service (Postgres + ClickHouse + web app) — self-host or cloud | Import & call (+ optional web UI) | Import & call |
| Multi-model broadcast + consensus voting | ✅ 5 strategies built in | ❌ | ❌ | ❌ | Partial — write your own scorer | ❌ |
| Meta-judge ("judge of judges") | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multi-step pipelines with gating/`run_if` | ✅ | ❌ (independent test cases) | ❌ | Via tracing + external orchestration | ✅ (task/solver chains) | ❌ |
| RAG-specific metrics | 3 built in | Extensive | ✅✅ core focus | Via integrations | Via custom scorers | Via prebuilt evaluators |
| Agent/trajectory evaluation | ✅ built-in judge type | ✅ | ❌ | Via tracing | ✅✅ core strength — 200+ evals | Via sibling `agentevals` package |
| LLM providers | Official SDKs (Anthropic, Gemini), extensible via `register_protocol` | Any, via LiteLLM/custom | Any, via LangChain LLM wrappers | Any — model-agnostic observability layer | OpenAI, Anthropic, Google, Groq, Mistral, xAI, Bedrock, Azure, local, ... | Any, via LangChain chat models |
| Config | Python or YAML | Python (pytest-style) | Python | Web UI + SDK | Python (`@task` decorators) | Python |
| License | MIT | Apache-2.0 | Apache-2.0 | MIT core (+ commercial enterprise add-ons) | MIT | MIT |

Read the row you care about, not the whole table: reaching for deep RAG
metrics specifically → Ragas is more mature there; broad agentic/safety
benchmarking → Inspect AI has far more built-in evals; you already run
Langfuse for tracing → its evals integrate with what you have. verdikt is
the pick when you want panel-of-judges consensus and gated pipelines as
first-class, typed Python, with nothing else to run.

*Comparison verified against each project's own docs/repo as of this
writing; these projects move fast, so if something's stale,
[open an issue](https://github.com/jatin-mishra/verdikt/issues).*

## Install

```bash
pip install -e .            # includes official provider SDKs (anthropic, google-genai)
pip install -e ".[dev]"     # + pytest
```

## Quick start

```python
import asyncio
from verdikt import Verdikt, JudgeConfig, EvalInput

vd = Verdikt(judges=[
    JudgeConfig(name="helpfulness", type="pointwise",
                model="anthropic/claude-haiku", threshold=0.7),
])
# API keys come from env vars: ANTHROPIC_AGENT_API_KEY, GEMINI_API_KEY

async def main():
    verdict = await vd.evaluate("helpfulness", EvalInput(
        input="What is the capital of France?",
        output="Paris is the capital of France.",
    ))
    print(verdict.score, verdict.passed, verdict.reasoning)

asyncio.run(main())
```

A fuller runnable version is at [`examples/quickstart.py`](examples/quickstart.py).

## Core concepts

| Type | What it is |
|---|---|
| `EvalInput` | What gets judged: `output` (required), plus optional `input`, `expected_output`, `context`, `conversation`, `trajectory`, `candidates` — each judge type validates the fields it needs. |
| `Verdict` | What comes back: `score` (0–1), `label`, `passed`, `reasoning`, `criteria_breakdown`, `confidence`, `sub_verdicts` (broadcast members), `error`, and `meta` (model, tokens, cost, latency, disagreement). |
| `JudgeConfig` | One judge's configuration: `type`, `model`, `criteria`/`labels`/`rubric`, `threshold`, `samples`, `llm_params` (provider-native passthrough — see [§4](#4-provider-native-params-the-systemuser-split-and-prompt-caching)), and an `execution` block for broadcast/judge-of-judges. |
| `PipelineConfig` | An ordered/parallel sequence of judges with gating (`on_fail`), conditions (`run_if`), and an aggregation strategy. |
| `Verdikt` | The facade: builds judges/pipelines from config, owns the `LLMClient`, and exposes `evaluate` / `evaluate_sync` / `evaluate_batch`. |

## YAML configuration

A complete reference config covering every judge type, execution mode, and pipeline lives at [`examples/verdikt.example.yaml`](examples/verdikt.example.yaml) (kept valid by the test suite); a runnable script is at [`examples/quickstart.py`](examples/quickstart.py). Condensed version:

```yaml
# verdikt.yaml
providers:
  anthropic: {api_key: ${ANTHROPIC_AGENT_API_KEY}}
  gemini:    {api_key: ${GEMINI_API_KEY}}

cache_path: .verdikt_cache.json        # optional: never pay twice for the same call

judges:
  - name: safety
    type: classifier
    model: anthropic/claude-haiku
    labels: [safe, unsafe]
    fail_on: [unsafe]

  - name: quality_panel            # broadcast to 3 models, meta-judge decides
    type: rubric
    criteria:
      - "Answers the actual question asked"
      - "No factual errors given the context"
    threshold: 0.7
    execution:
      mode: judge_of_judges
      models: [anthropic/claude-sonnet-4-5, anthropic/claude-haiku, gemini/gemini-2.5-pro]
      meta_judge: anthropic/claude-sonnet-4-5

pipelines:
  - name: support_bot_eval
    steps:
      - judge: safety
        on_fail: stop              # don't spend tokens on unsafe output
      - judge: quality_panel       # sees safety's verdict via prior_verdicts
    aggregation: all_pass
```

```python
vd = Verdikt.from_yaml("verdikt.yaml")
result = await vd.evaluate("support_bot_eval", EvalInput(input=q, output=a))
if not result.passed:
    agent.retry(feedback=result.verdicts[-1].reasoning)
```

Batch evaluation over a dataset (bounded concurrency; per-item failures become
error verdicts, never exceptions):

```python
batch = await vd.evaluate_batch("quality_panel", items, concurrency=8)
print(batch.failed_indices)
```

## Cookbook — every judge type

All judges accept the same `EvalInput` and return the same `Verdict`. Each example below shows the config (YAML) and the call.

### 1. `pointwise` — score one output

```yaml
- name: helpfulness
  type: pointwise
  model: anthropic/claude-haiku
  criteria: ["Directly answers the question", "No factual errors"]  # optional
  scale: {min: 1, max: 5}
  threshold: 0.7          # verdict.passed = score >= 0.7
  samples: 3              # optional: 3 runs, median wins (smooths noise)
```

```python
v = await vd.evaluate("helpfulness", EvalInput(input="What is DNS?", output=answer))
v.score       # 0.75  (normalized 0-1)
v.passed      # True
v.reasoning   # the judge's chain-of-thought
```

### 2. `reference` — compare against a gold answer

```yaml
- name: correctness
  type: reference
  model: anthropic/claude-sonnet-4-5
  threshold: 0.8
```

```python
v = await vd.evaluate("correctness", EvalInput(
    input="Capital of Australia?",
    output="It's Canberra.",
    expected_output="Canberra",     # required for this type
))
```

### 3. `pairwise` — A vs B comparison (A/B tests, model comparisons)

```yaml
- name: ab_test
  type: pairwise
  model: gemini/gemini-2.5-pro
  criteria: ["More helpful", "More accurate"]
  position_swap: true     # default: runs both orders, disagreement -> tie
```

```python
v = await vd.evaluate("ab_test", EvalInput(
    input="Explain HTTP caching",
    output="-",                              # ignored for pairwise
    candidates=[response_a, response_b],     # exactly 2
))
v.label   # "A" | "B" | "tie"
v.score   # 1.0 / 0.0 / 0.5
```

### 4. `rubric` — G-Eval style, per-criterion grading

```yaml
- name: support_quality
  type: rubric
  model: gemini/gemini-2.5-pro
  criteria:
    - "Resolves the customer's actual problem"
    - "Tone is professional and empathetic"
    - "No policy violations"
  threshold: 0.7
```

```python
v = await vd.evaluate("support_quality", EvalInput(input=ticket, output=reply))
v.score                                   # mean of criteria
for c in v.criteria_breakdown:            # per-criterion detail
    print(c.criterion, c.score, c.reasoning)
```

### 5. `classifier` — fixed label set (guardrails, routing)

```yaml
- name: safety
  type: classifier
  model: anthropic/claude-haiku
  labels: [safe, needs_review, unsafe]
  fail_on: [unsafe]        # verdict.passed = label not in fail_on
```

```python
v = await vd.evaluate("safety", EvalInput(output=agent_reply))
v.label    # "safe"
v.passed   # True
```

### 6–8. RAG judges — `rag_faithfulness`, `rag_context_relevance`, `rag_answer_relevance`

```yaml
- name: grounded          # is every claim supported by the context?
  type: rag_faithfulness
  model: anthropic/claude-haiku
  threshold: 0.8
- name: ctx_relevance     # was the retrieved context relevant to the question?
  type: rag_context_relevance
  model: anthropic/claude-haiku
- name: ans_relevance     # does the answer address the question?
  type: rag_answer_relevance
  model: anthropic/claude-haiku
```

```python
inp = EvalInput(
    input="What is our refund window?",
    output="Refunds are accepted within 30 days.",
    context=["Policy doc: customers may return items within 30 days..."],  # required for faithfulness/ctx_relevance
)
v = await vd.evaluate("grounded", inp)
```

### 9. `trajectory` — judge a full agent run

```yaml
- name: agent_run
  type: trajectory
  model: anthropic/claude-sonnet-4-5
  criteria: ["Task completed", "Tool calls correct and necessary", "No redundant loops"]
  threshold: 0.7
```

```python
from verdikt import Step

v = await vd.evaluate("agent_run", EvalInput(
    input="Book the cheapest flight BLR->DEL tomorrow",
    output="Booked 6E-204 at 7:10 for Rs 4,250.",
    trajectory=[
        Step(thought="search flights", tool="flight_search",
             tool_input={"from": "BLR", "to": "DEL"}, observation="6E-204 Rs4250; AI-501 Rs6100"),
        Step(thought="cheapest is 6E-204", tool="book",
             tool_input={"flight": "6E-204"}, observation="confirmed"),
    ],
))
```

### 10. Your own judge type

See [Writing a custom judge](#1-writing-a-custom-judge) below for a full worked example.

## Cookbook — every execution mode

Execution modes apply to **any** judge type (subject to the compatibility matrix below).

### `single` — one model (default)

```yaml
- name: quick_check
  type: pointwise
  model: anthropic/claude-haiku      # execution block omitted -> single
```

### `broadcast` + `average` — mean score across models

```yaml
- name: quality_panel
  type: pointwise
  execution:
    mode: broadcast
    models: [anthropic/claude-sonnet-4-5, anthropic/claude-haiku, gemini/gemini-2.5-pro]
    consensus: average             # default for score judges
```

```python
v = await vd.evaluate("quality_panel", inp)
v.score                  # mean of the 3 models
v.meta.disagreement      # 0-1; high = models disagree, worth human review
v.sub_verdicts           # each model's individual verdict + reasoning
```

### `broadcast` + `weighted_average` — trust some models more

```yaml
  execution:
    mode: broadcast
    models: [anthropic/claude-sonnet-4-5, gemini/gemini-2.5-pro]
    consensus: weighted_average
    weights: {"anthropic/claude-sonnet-4-5": 3.0, "gemini/gemini-2.5-pro": 1.0}
```

### `broadcast` + `majority_vote` — label/comparison judges

```yaml
- name: safety_jury
  type: classifier
  labels: [safe, unsafe]
  fail_on: [unsafe]
  execution:
    mode: broadcast
    models: [anthropic/claude-haiku, anthropic/claude-sonnet-4-5, gemini/gemini-2.5-flash]
    # consensus omitted -> majority_vote (the default for classifier/pairwise)
```

### `broadcast` + `unanimous` — strict gates (all models must agree)

```yaml
- name: strict_safety
  type: classifier
  labels: [safe, unsafe]
  fail_on: [unsafe]
  execution:
    mode: broadcast
    models: [anthropic/claude-sonnet-4-5, gemini/gemini-2.5-pro]
    consensus: unanimous          # any split verdict -> passed = false
```

### `broadcast` + `consensus_leader` — one model decides, others sanity-check

```yaml
  execution:
    mode: broadcast
    models: [anthropic/claude-sonnet-4-5, anthropic/claude-haiku, gemini/gemini-2.5-pro]
    consensus: consensus_leader
    leader: anthropic/claude-sonnet-4-5   # defaults to first model if omitted
    on_disagreement: accept_leader        # or: fail | escalate | accept_consensus
    disagreement_threshold: 0.25
```

`on_disagreement` fires when `meta.disagreement` exceeds the threshold: `accept_leader` keeps the leader's verdict, `fail` forces `passed=false`, `escalate` flags the verdict (`meta.extra["needs_escalation"]`) for human review.

### `judge_of_judges` — a meta-judge weighs arguments, not just votes

```yaml
- name: final_review
  type: rubric
  criteria: ["Factually correct", "Complete"]
  threshold: 0.7
  execution:
    mode: judge_of_judges
    models: [anthropic/claude-sonnet-4-5, anthropic/claude-haiku, gemini/gemini-2.5-pro]
    meta_judge: anthropic/claude-sonnet-4-5
```

```python
v = await vd.evaluate("final_review", inp)
v.reasoning       # the meta-judge's explanation of how it weighed each judge
v.sub_verdicts    # the 3 member verdicts it read (scores AND reasoning)
```

Better than voting when models disagree *for different reasons* — the meta-judge reads each member's reasoning and discounts weak or generic arguments.

### Multi-sample voting — works in any mode

```yaml
- name: stable_score
  type: pointwise
  model: anthropic/claude-haiku
  samples: 5        # 5 runs per model; median score / majority label
```

In broadcast mode, `samples` applies per member model (3 models × 5 samples = 15 calls).

## Integration cookbook — plugging verdikt into agents

verdikt has no framework dependency: anywhere you can call an async (or sync) function, you can judge an output. All examples share this setup:

```python
from verdikt import Verdikt, EvalInput

vd = Verdikt.from_yaml("verdikt.yaml")
```

### 1. Quality gate in a plain agent loop (judge -> retry with feedback)

```python
async def answer_with_gate(question: str, max_attempts: int = 3) -> str:
    feedback = ""
    for _ in range(max_attempts):
        reply = await my_agent.run(question + feedback)
        v = await vd.evaluate("helpfulness", EvalInput(input=question, output=reply))
        if v.passed:
            return reply
        # verdict reasoning doubles as a self-correction signal
        feedback = f"\n\nYour previous answer was rejected: {v.reasoning}. Fix it."
    return reply  # best effort after retries
```

### 2. Safety gate before sending anything to the user

```python
reply = await my_agent.run(user_msg)
v = await vd.evaluate("safety", EvalInput(output=reply))
if not v.passed:
    reply = "Sorry, I can't help with that."
```

### 3. Full pipeline as a post-processing step (gates + panel + prior verdicts)

```python
result = await vd.evaluate("support_bot_eval", EvalInput(
    input=user_msg, output=reply, context=retrieved_docs))
if result.stopped_early:              # failed the safety gate, rest was skipped
    escalate_to_human(user_msg, reply, result.verdicts[0].reasoning)
elif not result.passed:
    reply = await my_agent.rewrite(reply, feedback=result.reasoning)
```

### 4. Sync codebase (no async)

```python
v = vd.evaluate_sync("helpfulness", EvalInput(input=q, output=a))
```

### 5. RAG pipeline — judge groundedness right after generation

```python
docs = retriever.search(question)
answer = await llm.generate(question, docs)
v = await vd.evaluate("grounded", EvalInput(input=question, output=answer, context=docs))
if not v.passed:                       # hallucination detected
    answer = await llm.generate(question, docs, strict_grounding=True)
```

### 6. Tool-using agent — judge the whole trajectory, not just the answer

```python
from verdikt import Step

steps = []
for action in agent_run.actions:       # however your framework exposes them
    steps.append(Step(thought=action.thought, tool=action.tool,
                      tool_input=action.args, observation=str(action.result)))

v = await vd.evaluate("agent_run", EvalInput(
    input=task, output=agent_run.final_answer, trajectory=steps))
```

### 7. LangGraph — judge as a node with conditional routing

```python
async def judge_node(state):
    v = await vd.evaluate("helpfulness", EvalInput(
        input=state["question"], output=state["draft"]))
    return {"verdict": v}

def route(state):
    return "publish" if state["verdict"].passed else "revise"

graph.add_node("judge", judge_node)
graph.add_conditional_edges("judge", route, {"publish": "publish", "revise": "generate"})
```

### 8. Multi-turn chat — judge with conversation history

```python
from verdikt import Message

v = await vd.evaluate("helpfulness", EvalInput(
    output=latest_reply,
    conversation=[Message(role=m["role"], content=m["content"]) for m in history],
))
```

### 9. Offline dataset evaluation (nightly evals, before a release)

```python
items = [EvalInput(input=r["question"], output=r["answer"],
                   expected_output=r["gold"]) for r in dataset]
batch = await vd.evaluate_batch("correctness", items, concurrency=8)
pass_rate = 1 - len(batch.failed_indices) / batch.total
for i in batch.failed_indices:
    print(dataset[i]["question"], "->", batch.results[i].reasoning)
```

### 10. CI regression test (pytest) — block merges that degrade quality

```python
import pytest
from verdikt import Verdikt, EvalInput

vd = Verdikt.from_yaml("verdikt.yaml")

@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_answer_quality(case):
    answer = my_agent.run_sync(case["question"])
    v = vd.evaluate_sync("correctness", EvalInput(
        input=case["question"], output=answer, expected_output=case["gold"]))
    assert v.passed, v.reasoning
```

### 11. A/B testing a prompt or model change (pairwise)

```python
wins = {"A": 0, "B": 0, "tie": 0}
for q in sample_questions:
    old = await agent_v1.run(q)
    new = await agent_v2.run(q)
    v = await vd.evaluate("ab_test", EvalInput(input=q, output="-", candidates=[old, new]))
    wins[v.label] += 1
print(wins)   # ship v2 only if it actually wins
```

### 12. Routing on judge disagreement (human-in-the-loop)

```python
v = await vd.evaluate("quality_panel", inp)      # broadcast judge
if v.meta.disagreement and v.meta.disagreement > 0.4:
    queue_for_human_review(inp, v.sub_verdicts)  # models can't agree -> human decides
```

## Judge type × execution mode compatibility

Not every consensus strategy makes sense for every judge type — averaging
A/B/tie outcomes or class labels is meaningless. The library enforces this at
construction time (bad configs fail fast with the valid options listed) and
picks a sensible default consensus per judge type when you don't set one:

| Judge type | Verdict | single | broadcast consensus | judge_of_judges | default consensus |
|---|---|---|---|---|---|
| pointwise, reference, rag_*, trajectory | score | ✓ | average, weighted_average, majority_vote, unanimous, consensus_leader | ✓ | average |
| rubric | score + per-criterion | ✓ | same as above; per-criterion scores are merged across models | ✓ | average |
| classifier | label | ✓ | majority_vote, unanimous, consensus_leader only | ✓ | majority_vote |
| pairwise | A/B/tie | ✓ | majority_vote, unanimous, consensus_leader only (each model runs its own position swap) | ✓ | majority_vote |

Custom judges declare their own compatibility via class attributes:
`supported_modes`, `allowed_consensus`, `default_consensus`.

## Extending verdikt

verdikt is meant to be *embedded* — dropped into your own server, agent
framework, or eval harness and grown from there. Every piece below is a
small interface you implement once and register; nothing requires editing
verdikt's own source.

| Want to... | Implement | Register with |
|---|---|---|
| Add a new kind of judge (custom scoring/labeling logic) | `BaseJudge` subclass | `@register("my_type")` |
| Point at an LLM backend verdikt doesn't wire up (an internal gateway, a local model server, a mocked test double) | `LLMClient` subclass | pass as `client=` |
| Add a wire protocol / provider SDK verdikt doesn't ship | `ProtocolAdapter` subclass | `register_protocol(...)` |
| Ship a reusable prompt template with your judge | a `.j2` file | `add_template_dir(...)` |
| Set `top_p`/`stop_sequences`/thinking budgets/caching/whatever a provider adds next | nothing — it's data | `JudgeConfig(llm_params={...})` |

### 1. Writing a custom judge

A judge is `BaseJudge.parse()` (turn the model's JSON reply into a `Verdict`)
plus a prompt template. Here's a complete one that flags PII leaked into an
AI's output — a type not built in:

```python
# my_judges.py
from pathlib import Path
from typing import Any

from verdikt import BaseJudge, EvalInput, Verdict, register, add_template_dir

# ship pii_check.j2 next to this file; searched before the built-in templates
add_template_dir(Path(__file__).parent / "templates")


@register("pii_check")
class PIIJudge(BaseJudge):
    system_template = None  # simple judge: skip the system/user split (see below), one message
    user_template = "pii_check.j2"
    verdict_type = "label"

    # labels can't be averaged: only vote-style consensus makes sense
    allowed_consensus = ("majority_vote", "unanimous", "consensus_leader")
    default_consensus = "majority_vote"

    def parse(self, data: dict[str, Any], inp: EvalInput) -> Verdict:
        leaked = data.get("categories") or []
        return Verdict(
            judge_name=self.name,
            label="unsafe" if leaked else "safe",
            reasoning=str(data.get("reasoning", "")),
        )
```

```jinja2
{# templates/pii_check.j2 #}
Scan the AI response below for leaked personal data (emails, phone numbers,
SSNs, card numbers). List every category you find.

AI response:
{{ output }}

Respond with ONLY this JSON:
{"reasoning": "<what you found, if anything>", "categories": ["<category>", ...]}
```

```python
import my_judges  # runs @register("pii_check") + add_template_dir on import

vd = Verdikt(judges=[
    JudgeConfig(name="pii", type="pii_check", model="anthropic/claude-haiku",
                fail_on=["unsafe"]),
])
v = await vd.evaluate("pii", EvalInput(output=agent_reply))
```

Skip the template file entirely for smaller judges: reuse a built-in
(`user_template = "pointwise_user.j2"`) or override per-config with
`JudgeConfig(prompt_template="...")` — a raw Jinja2 string, no file needed,
sent as a single message. `ScoreParsingMixin` (`from verdikt.core.base import
ScoreParsingMixin`) gives you `parse()` for free for `{reasoning, score,
confidence}`-shaped replies.

Override `required_fields()` to validate `EvalInput` up front (default:
`["output"]`), or `template_context()` to add extra variables your template
needs. The judge type name (`"pii_check"` above) is what your YAML `type:`
field or `JudgeConfig(type=...)` refers to — same mechanism, code or config.

### 2. Writing a custom LLM client

`LLMClient` is one async method. Point verdikt at an internal gateway, a
local model server, or anything else that can turn `(model, messages)` into
text:

```python
from verdikt import LLMClient, LLMResponse

class MyGatewayClient(LLMClient):
    async def complete(self, model, messages, **kw) -> LLMResponse:
        text = await my_internal_gateway.chat(model, messages)
        return LLMResponse(text=text, model=model)

vd = Verdikt.from_config(cfg, client=MyGatewayClient())
```

This composes with the built-in reliability wrappers: wrap your client in
`RetryingClient` (backoff + fallback models) or `CachingClient` (skip
duplicate calls) exactly like `FrontierClient` does — see
`verdikt/facade.py::_default_client`.

### 3. Plugging in a new provider protocol

If you'd rather stay on `FrontierClient`'s retry/broadcast/caching plumbing
but need a provider whose wire protocol verdikt doesn't speak (a different
SDK, a custom gateway), implement `ProtocolAdapter` — the same interface
`AnthropicAdapter`/`GeminiAdapter` use internally — and register it:

```python
from verdikt import ProtocolAdapter, register_protocol, Verdikt, ProviderConfig

class MyProtocolAdapter(ProtocolAdapter):
    async def complete(self, base_url, api_key, model, messages, temperature,
                        max_tokens, json_mode, params, *, timeout, transport=None):
        text = await my_sdk_client(api_key, base_url).chat(model, messages)
        return text, input_tokens, output_tokens   # your SDK's own types stop here

register_protocol("myproto", MyProtocolAdapter, providers={"myprovider": "https://api.example.com"})

vd = Verdikt(
    judges=[...],
    providers={"myprovider": ProviderConfig(api_key="...")},  # model: "myprovider/..."
)
```

Only `MyProtocolAdapter` ever sees your SDK's request/response objects —
`FrontierClient` only sees the `(text, input_tokens, output_tokens)` tuple,
same contract every built-in adapter follows. Register before constructing
`Verdikt`/`FrontierClient` (adapters resolve lazily on first use, so
importing the module that calls `register_protocol` early is enough).

### 4. Provider-native params, the system/user split, and prompt caching

Every judge's prompt is two messages, not one: `system_template` renders
whatever comes from `JudgeConfig` (criteria, rubric, labels, scale,
few_shot) — identical on every call to that judge — and `user_template`
renders whatever comes from `EvalInput` (input, output, context,
conversation, trajectory, ...) — different every call. This isn't just
tidiness: it's what lets `AnthropicAdapter`/`GeminiAdapter` put your judge's
instructions in the provider's actual `system` field instead of burying
them inside a "user" message, which both models follow better, and — since
the system half is byte-identical across repeated calls to the same judge —
it's exactly the shape prompt caching wants.

Anything beyond `temperature`/`max_tokens` — `top_p`, `top_k`,
`stop_sequences`, extended-thinking budgets, tool use, safety settings,
whatever a provider adds next — goes on `JudgeConfig.llm_params` and reaches
the SDK call as-is. Each `ProtocolAdapter` decides what it understands;
nothing above it needs to change when a provider ships something new:

```python
JudgeConfig(
    name="strict_check", type="pointwise", model="anthropic/claude-sonnet-4-5",
    llm_params={
        "top_p": 0.9,
        "stop_sequences": ["<END>"],
        "cache_system_prompt": True,  # Anthropic only: mark the system block
                                       # cacheable (AnthropicAdapter pops this
                                       # before calling messages.create(); it's
                                       # a verdikt convenience name, not a raw
                                       # Anthropic param). GeminiAdapter drops
                                       # it silently if the same llm_params is
                                       # reused on a gemini/* broadcast member.
    },
)
```

Writing your own `ProtocolAdapter` (§3 above)? Do the same thing: pop any
convenience flags you invent out of `params` before spreading the rest into
your SDK call, exactly like `AnthropicAdapter.complete()` does.

### A note on consensus strategies

Consensus strategies (`average`, `majority_vote`, ...) are intentionally a
fixed, validated set — see [Judge type × execution mode
compatibility](#judge-type--execution-mode-compatibility) for why some
combinations are rejected outright (averaging labels isn't meaningful). For
a custom aggregation rule, run `broadcast` mode and combine
`verdict.sub_verdicts` yourself — each member's score, label, and reasoning
are all there.

## Architecture at a glance

```
EvalInput ─▶ BaseJudge.parse()+template ─▶ LLMClient.complete() ─▶ raw JSON ─▶ Verdict
                    │                              │
             @register("type")            RetryingClient ▸ CachingClient ▸ FrontierClient
                    │                              │
              Executor (single /                ProtocolAdapter (per provider SDK)
              broadcast / judge_of_judges)         │
                    │                       register_protocol(...)
            PipelineRunner (multi-step,
            gating, prior_verdicts)
```

Every layer only depends on the interface below it, not the implementation:
`Executor` doesn't know which judge type it's running, `LLMClient` callers
don't know which provider (or SDK) sits behind it, and `PipelineRunner`
doesn't know what any individual judge does internally. That's what makes
each layer independently swappable — write a judge without touching the LLM
layer, or a backend without touching judges or pipelines.

## Debugging LLM calls

`FrontierClient` prints the exact request going out (system prompt, every message, temperature/max_tokens/json_mode) and the exact response coming back (text, tokens, cost, latency) to the console, in bordered/colored blocks. It's on by default for now:

```bash
export VERDIKT_LOG_LLM_CALLS=0   # turn it off
```

```python
from verdikt.llm.logging import set_llm_logging

set_llm_logging(False)   # or True — overrides the env var for this process
```

`Verdikt(...)` also accepts `client=FrontierClient(log_calls=False)` to control it per client instance.

## Tests

```bash
python -m pytest tests -q   # 65 tests, no network or API keys needed
```

## Contributing

Bug reports, feature requests, and PRs are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the test/lint commands PRs
are expected to pass, and the release process. This project follows the
[Contributor Covenant](CODE_OF_CONDUCT.md). For security issues, see
[SECURITY.md](SECURITY.md) rather than opening a public issue.

## License

MIT — see [LICENSE](LICENSE).
