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

- [x] 19. REPLACED raw-httpx FrontierClient adapters with official provider
      SDKs (`anthropic`, `google-genai`), and — per user request — DROPPED
      every provider except Anthropic and Gemini (OpenAI, Kimi, Mistral,
      OpenRouter, xAI, DeepSeek, and the generic OpenAI-compatible protocol
      removed entirely from `ProviderRegistry`, cost table, adapters, tests,
      README, and the example YAML). Each adapter now owns its provider's SDK
      request/response types exclusively (`adapters/{anthropic,gemini}.py`);
      `FrontierClient`/`ProtocolAdapter` only ever see the generic
      `(text, input_tokens, output_tokens)` tuple. SDK clients are cached per
      `(base_url, api_key)` on each adapter instance (`ProtocolAdapter.
      _client_for`), `max_retries=0` on the SDK clients so retries stay
      centralized in `RetryingClient`, and `httpx.MockTransport` injection
      (`http_client=` / `HttpOptions(async_client_args={"transport": ...})`)
      keeps the existing wire-level test style working end-to-end through the
      real SDKs. Added `verdikt/llm/logging.py`: a feature-flagged
      (`VERDIKT_LOG_LLM_CALLS`, on by default), colorized console logger of
      the exact outgoing request (system prompt, messages, params) and exact
      incoming response (text, tokens, cost, latency), hooked into
      `FrontierClient.complete()` as the single choke-point every provider
      call passes through. 57 tests passing.

- [x] 20. README rewrite (professional structure, TOC, "Extending verdikt"
      section) + new public extension points: `register_protocol` (plug in a
      custom `ProtocolAdapter`, `verdikt/llm/frontier/adapters/__init__.py`)
      and `add_template_dir` (custom `.j2` template directories,
      `verdikt/prompts/__init__.py`). Fixed a real gap found while writing
      this: `EvalInput.system_prompt`/`.conversation` were declared on the
      schema but never rendered into any judge's prompt — wired both into
      `BaseJudge.template_context()` + new `system_prompt`/`conversation`
      macros in `_macros.j2`, applied across all 6 live-response judge
      templates. `EvalInput.metadata` deliberately left un-rendered (caller
      bookkeeping only, documented as such). `examples/quickstart.py`
      rewritten into 41 numbered, independently-runnable `case_*` functions
      covering every EvalInput/JudgeConfig/ExecutionConfig/PipelineConfig
      field, every judge type, every consensus/aggregation strategy, every
      extension point, and 2 production-style multi-feature cases — all 41
      verified to actually run against real Anthropic/Gemini calls. 60 tests
      passing.

- [x] 21. Open-source readiness: `LICENSE` (MIT), `CONTRIBUTING.md` (dev
      setup, release process incl. one-time PyPI Trusted Publisher setup),
      `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`,
      `CHANGELOG.md`. `pyproject.toml` gained authors/keywords/classifiers/
      urls + a `[tool.ruff]` config; `verdikt/py.typed` added (PEP 561).
      Verified (not assumed) that hatchling's default wheel build actually
      includes `prompts/templates/*.j2` and `py.typed` by building the wheel
      and pip-installing it into a throwaway venv. Ran `ruff check .`,
      fixed all real findings (ambiguous variable name, missing `zip(...,
      strict=True)`, stale `Union`/`Optional` typing imports across most of
      the package — safe given `from __future__ import annotations`
      everywhere). Added `.github/workflows/ci.yml` (test matrix, py3.10–
      3.13 + ruff) and `publish.yml` (PyPI publish on GitHub Release, via
      OIDC Trusted Publishing — no token secret), plus issue/PR templates.
      README gained a badges row and a researched, sourced "How it
      compares" table (verdikt vs DeepEval/Ragas/Langfuse/Inspect
      AI/OpenEvals). Did NOT push to GitHub or publish to PyPI — that's
      left for the user to trigger deliberately.

- [x] 22. User flagged a real gap: `BaseJudge` only ever forwarded
      `temperature`/`max_tokens` to `client.complete()` — no per-judge way to
      set `top_p`/`stop_sequences`/thinking budgets/etc, and every judge's
      own instructional prompt was jammed into a single "user" message,
      never using Anthropic's/Gemini's native `system` field even though
      each adapter already knew how to route one. Fixed both:
      - `JudgeConfig.llm_params: dict[str, Any]` (`core/schemas.py`) —
        provider-native passthrough forwarded from `judge_once()` and
        `_meta_judge()` straight to `client.complete()`; unknown keys reach
        the SDK call as-is, so a new provider feature never needs a verdikt
        code change (each `ProtocolAdapter` decides what it understands).
      - Split every judge's prompt into two templates:
        `BaseJudge.system_template` (renders JudgeConfig-derived,
        call-invariant content — criteria/rubric/labels/scale/few_shot) and
        `user_template` (renders EvalInput-derived, per-call content —
        input/output/context/conversation/trajectory/priors). Partition is a
        fixed key set (`_SYSTEM_KEYS` in `core/base.py`), so existing
        `template_context()` overrides (RAG's `metric`, pairwise's
        `candidate_a/b`) needed no changes. `build_messages()` replaces the
        old `build_prompt()`; `system_template = None` opts a judge out of
        the split (single message, e.g. simple custom judges). Renamed all
        7 template pairs (`{type}_system.j2`/`{type}_user.j2`,
        `meta_judge_*.j2` too) — deleted the 7 old combined files.
      - Bonus this unlocked cheaply: `AnthropicAdapter` now supports a
        `cache_system_prompt: True` convenience flag in `llm_params` that
        wraps the system message in Anthropic's `cache_control: ephemeral`
        block — since that block is now genuinely identical across every
        call to a given judge, it's a real caching win, not just a shape
        change. `GeminiAdapter` defensively pops/ignores the same flag so a
        judge broadcasting shared `llm_params` to both providers doesn't
        crash on Gemini (no equivalent inline mechanism there — Gemini's
        context caching is a separate, heavier API, left as a future item).
      - Fixed 5 test sites across `test_pipeline.py`/`test_config_and_facade.py`/
        `test_example_yaml.py`/`test_execution.py` that inspected
        `messages[-1]["content"]` to route fake responses — "Allowed
        labels"/"meta-judge" moved to the now-separate system message, so
        these now scan the whole conversation (`conftest.py`'s new
        `contains()` helper). Verified end-to-end against the real Anthropic
        API (system/user split visible in the debug log, `cache_control`
        accepted, `stop_sequences` honored) and confirmed Gemini doesn't
        error on the shared `cache_system_prompt` flag. 65 tests passing.

## Status: COMPLETE (v0.1.0, unpublished)

All planned work is done and delivered. If resuming for follow-up work, run
`python3 -m pytest tests -q` first to confirm state.

## Possible next steps (not requested yet)

- Calibration tooling (`verdikt calibrate` vs a human-labeled set)
- Escalation execution for on_disagreement=escalate (currently flags meta.extra["needs_escalation"])
- Sync batch wrapper; streaming progress callbacks
- mypy in CI (codebase isn't mypy-clean end-to-end yet; deliberately not
  bundled into the ruff/CI work in #21 — separate effort)
- Actually push to GitHub + cut the first PyPI release (see CONTRIBUTING.md
  "Releasing")
- Gemini context caching (a separate, heavier API than Anthropic's inline
  cache_control -- requires creating a CachedContent resource ahead of time)
