# verdikt

Configurable LLM-as-judge library. Plug it into any agent: configure judges (in code or YAML), feed the AI's input/output, get a structured judgement back.

**Features**

- 9 built-in judge types: `pointwise`, `reference`, `pairwise` (with position-swap), `rubric` (G-Eval style), `classifier`, `rag_faithfulness`, `rag_context_relevance`, `rag_answer_relevance`, `trajectory` (agent runs) — plus custom judges via `@register`.
- Execution modes per judge: `single` model, `broadcast` to many models (GPT, Claude, Gemini, Kimi, ...) with consensus (`majority_vote`, `average`, `weighted_average`, `unanimous`, `consensus_leader`), or `judge_of_judges` (a meta-judge weighs all verdicts and reasoning).
- Multi-step pipelines: sequential/parallel steps, early-exit gates (`on_fail: stop`), conditional steps (`run_if`), and automatic `prior_verdicts` passing so later judges see earlier verdicts.
- Reliability built in: reasoning-before-score prompts, JSON-only outputs, multi-sample voting, retries with backoff, model fallbacks, optional response caching, injection-resistant prompt delimiting, cost/token tracking on every verdict.

## Install

```bash
pip install -e .            # includes native frontier-provider client (httpx)
pip install -e ".[dev]"     # + pytest
```

## Quick start (code only)

```python
import asyncio
from verdikt import Verdikt, JudgeConfig, EvalInput

v = Verdikt(judges=[
    JudgeConfig(name="helpfulness", type="pointwise",
                model="openai/gpt-4.1-mini", threshold=0.7),
])
# API keys come from env vars: OPENAI_API_KEY, ANTHROPIC_API_KEY,
# GEMINI_API_KEY, MOONSHOT_API_KEY (Kimi), ...

async def main():
    verdict = await v.evaluate("helpfulness", EvalInput(
        input="What is the capital of France?",
        output="Paris is the capital of France.",
    ))
    print(verdict.score, verdict.passed, verdict.reasoning)

asyncio.run(main())
```

## YAML configuration

A complete reference config covering every judge type, execution mode, and pipeline lives at [`examples/verdikt.example.yaml`](examples/verdikt.example.yaml) (kept valid by the test suite); a runnable script is at [`examples/quickstart.py`](examples/quickstart.py). Condensed version:

```yaml
# verdikt.yaml
providers:
  openai:    {api_key: ${OPENAI_API_KEY}}
  anthropic: {api_key: ${ANTHROPIC_API_KEY}}
  gemini:    {api_key: ${GEMINI_API_KEY}}
  kimi:      {api_key: ${MOONSHOT_API_KEY}, base_url: "https://api.moonshot.ai/v1"}

cache_path: .verdikt_cache.json        # optional: never pay twice for the same call

judges:
  - name: safety
    type: classifier
    model: openai/gpt-4.1-mini
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
      models: [openai/gpt-4.1, anthropic/claude-sonnet, kimi/kimi-k3]
      meta_judge: anthropic/claude-sonnet

pipelines:
  - name: support_bot_eval
    steps:
      - judge: safety
        on_fail: stop              # don't spend tokens on unsafe output
      - judge: quality_panel       # sees safety's verdict via prior_verdicts
    aggregation: all_pass
```

```python
v = Verdikt.from_yaml("verdikt.yaml")
result = await v.evaluate("support_bot_eval", EvalInput(input=q, output=a))
if not result.passed:
    agent.retry(feedback=result.verdicts[-1].reasoning)
```

Batch evaluation over a dataset (bounded concurrency; per-item failures become
error verdicts, never exceptions):

```python
batch = await v.evaluate_batch("quality_panel", items, concurrency=8)
print(batch.failed_indices)
```

## Cookbook — every judge type

All judges accept the same `EvalInput` and return the same `Verdict`. Each example below shows the config (YAML) and the call.

### 1. `pointwise` — score one output

```yaml
- name: helpfulness
  type: pointwise
  model: openai/gpt-4.1-mini
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
  model: openai/gpt-4.1
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
  model: openai/gpt-4.1-mini
  labels: [safe, needs_review, unsafe]
  fail_on: [unsafe]        # verdict.passed = label not in fail_on
```

```python
v = await vd.evaluate("safety", EvalInput(output=agent_reply))
v.label    # "safe"
v.passed   # True
```

### 6-8. RAG judges — `rag_faithfulness`, `rag_context_relevance`, `rag_answer_relevance`

```yaml
- name: grounded          # is every claim supported by the context?
  type: rag_faithfulness
  model: openai/gpt-4.1-mini
  threshold: 0.8
- name: ctx_relevance     # was the retrieved context relevant to the question?
  type: rag_context_relevance
  model: openai/gpt-4.1-mini
- name: ans_relevance     # does the answer address the question?
  type: rag_answer_relevance
  model: openai/gpt-4.1-mini
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

### 10. Custom judge type

```python
from verdikt import BaseJudge, register
from verdikt.core.base import ScoreParsingMixin

@register("tone")
class ToneJudge(ScoreParsingMixin, BaseJudge):
    template = "pointwise.j2"                      # reuse a built-in template
    allowed_consensus = ("average", "majority_vote")  # optional: own compatibility rules
# then in YAML: {name: tone_check, type: tone, model: openai/gpt-4.1-mini}
```

## Cookbook — every execution mode

Execution modes apply to **any** judge type (subject to the compatibility matrix below).

### `single` — one model (default)

```yaml
- name: quick_check
  type: pointwise
  model: openai/gpt-4.1-mini      # execution block omitted -> single
```

### `broadcast` + `average` — mean score across models

```yaml
- name: quality_panel
  type: pointwise
  execution:
    mode: broadcast
    models: [openai/gpt-4.1, anthropic/claude-sonnet-4-5, kimi/kimi-k3]
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
    models: [openai/gpt-4.1, kimi/kimi-k3]
    consensus: weighted_average
    weights: {"openai/gpt-4.1": 3.0, "kimi/kimi-k3": 1.0}
```

### `broadcast` + `majority_vote` — label/comparison judges

```yaml
- name: safety_jury
  type: classifier
  labels: [safe, unsafe]
  fail_on: [unsafe]
  execution:
    mode: broadcast
    models: [openai/gpt-4.1-mini, anthropic/claude-haiku, gemini/gemini-2.5-flash]
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
    models: [openai/gpt-4.1, anthropic/claude-sonnet-4-5]
    consensus: unanimous          # any split verdict -> passed = false
```

### `broadcast` + `consensus_leader` — one model decides, others sanity-check

```yaml
  execution:
    mode: broadcast
    models: [anthropic/claude-sonnet-4-5, openai/gpt-4.1-mini, kimi/kimi-k3]
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
    models: [openai/gpt-4.1, anthropic/claude-sonnet-4-5, kimi/kimi-k3]
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
  model: openai/gpt-4.1-mini
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

### 13. Custom backend or on-prem model — swap the client

```python
from verdikt import LLMClient, LLMResponse

class MyGatewayClient(LLMClient):
    async def complete(self, model, messages, **kw) -> LLMResponse:
        text = await my_internal_gateway.chat(model, messages)
        return LLMResponse(text=text, model=model)

vd = Verdikt.from_config(cfg, client=MyGatewayClient())
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

## Custom judges

```python
from verdikt import BaseJudge, register
from verdikt.core.base import ScoreParsingMixin

@register("tone")
class ToneJudge(ScoreParsingMixin, BaseJudge):
    template = "pointwise.j2"          # or set config.prompt_template inline

# now usable in YAML: {name: tone_check, type: tone, model: ...}
```

Any backend works — implement `LLMClient.complete()` and pass it as `client=` to replace the built-in `FrontierClient`.

## Verdict shape

Every judge returns a `Verdict`: `score` (0–1), `label`, `passed`, `reasoning` (always present), `criteria_breakdown`, `confidence`, `sub_verdicts` (broadcast members), `error` (failures never raise mid-pipeline), and `meta` (model(s), tokens, cost, latency, disagreement score, execution mode).

## Tests

```bash
python -m pytest tests -q   # 58 tests, no network or API keys needed
```
