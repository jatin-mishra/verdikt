# verdikt — Implementation Progress Tracker

Purpose: if this session hits its quota, a new session can resume from here.

Design doc: `/home/claude/llm-judge-library-plan.md` (also delivered to user).
Project root: `/home/claude/verdikt`

## Checklist

- [x] 1. Scaffold: pyproject.toml, package skeleton, this tracker
- [x] 2. core/schemas.py — EvalInput, Verdict, CriterionResult, JudgeMeta, PipelineVerdict, configs
- [x] 3. llm/ — LLMClient, RetryingClient (backoff + fallbacks), FrontierClient (native httpx adapters: openai/anthropic/gemini protocols), ProviderRegistry + API keys, CachingClient, cost table
- [x] 4. prompts/ — Jinja2 templates: pointwise, pairwise, rubric, classifier, rag, trajectory, meta_judge, _macros
- [x] 5. core/base.py + registry — BaseJudge with multi-sample voting; @register
- [x] 6. judges/ — pointwise, reference, pairwise (position swap), rubric, classifier, rag x3, trajectory
- [x] 7. execution/ — Executor (single/broadcast/judge_of_judges), consensus strategies, disagreement score + actions
- [x] 8. pipeline/ — runner (gating, run_if, parallel, prior_verdicts), aggregation, BatchRunner
- [x] 9. config/loader.py — YAML + ${ENV} interpolation
- [x] 10. facade.py + __init__.py — Verdikt class, evaluate / evaluate_sync / evaluate_batch
- [x] 11. tests/ — 41 tests, FakeLLMClient, no network needed
- [x] 12. All tests passing (41 passed)
- [x] 13. README.md written
- [x] 14. Zip delivered to user

- [x] 15. Judge-type x execution-mode compatibility matrix: validation at Executor
      construction, per-type default consensus (majority_vote for classifier/pairwise),
      rubric criteria_breakdown merged across broadcast members
- [x] 16. REMOVED library-level quota checkpoint/resume per user clarification
      (the "resume on quota" requirement referred to THIS session tracker, not a
      library feature). Deleted: checkpoint.py, QuotaExceededError, PipelineInterrupted,
      run_id params, batch resume. Retries/fallbacks/cache kept. 48 tests passing.

- [x] 17. REPLACED litellm with native FrontierClient per user request: direct HTTP
      via httpx, three protocols (openai-compatible incl. Kimi/Mistral/xAI/DeepSeek/
      OpenRouter/vLLM/Ollama; anthropic messages; gemini generateContent).
      ProviderConfig gained optional `protocol` field. 56 tests passing.

- [x] 18. Added examples/ to source: verdikt.example.yaml (every judge type, execution
      mode, and pipeline; validated by test_example_yaml.py) + quickstart.py. 58 tests.

## Status: COMPLETE (v0.1.0)

All planned work is done and delivered. If resuming for follow-up work, run
`python3 -m pytest tests -q` from /home/claude/verdikt first to confirm state.

## Possible next steps (not requested yet)

- Calibration tooling (`verdikt calibrate` vs a human-labeled set)
- Escalation execution for on_disagreement=escalate (currently flags meta.extra["needs_escalation"])
- Sync batch wrapper; streaming progress callbacks
- PyPI packaging polish (CI, ruff, mypy)
