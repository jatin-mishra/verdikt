# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - not yet published

Initial feature set:

### Added

- 9 built-in judge types: `pointwise`, `reference`, `pairwise` (with
  position-swap), `rubric`, `classifier`, `rag_faithfulness`,
  `rag_context_relevance`, `rag_answer_relevance`, `trajectory` — plus
  custom judge types via `@register`.
- Execution modes per judge: `single`, `broadcast` (with `average`,
  `weighted_average`, `majority_vote`, `unanimous`, `consensus_leader`
  consensus), and `judge_of_judges`.
- Multi-step pipelines: sequential/parallel steps, `on_fail` gating,
  `run_if` conditions, automatic `prior_verdicts` passing, and
  `all_pass`/`weighted_average`/`majority_vote`/`last` aggregation.
- Reliability: reasoning-before-score prompts, JSON-only outputs,
  multi-sample voting, retries with backoff, cross-provider model
  fallbacks, optional response caching, injection-resistant prompt
  delimiting (`_macros.j2`), cost/token tracking on every verdict.
- Native LLM backends for Anthropic and Google Gemini via their official
  Python SDKs (`anthropic`, `google-genai`) — each provider's SDK
  request/response types are owned entirely by its own adapter
  (`verdikt/llm/frontier/adapters/`).
- Extension points: `@register` for judge types, `LLMClient` for custom
  backends, `register_protocol` for custom wire protocols/SDKs,
  `add_template_dir` for custom prompt templates.
- `EvalInput.system_prompt` and `.conversation` are rendered into every
  judge's prompt (multi-turn/system-prompt-aware judging).
- Every judge's prompt is split into a `system` message (JudgeConfig-derived,
  identical across calls — criteria/rubric/labels/scale/few_shot) and a
  `user` message (EvalInput-derived, per-call) so providers' native system
  slot is actually used (`BaseJudge.system_template`/`user_template`).
  `JudgeConfig.llm_params` forwards arbitrary provider-native parameters
  (`top_p`, `stop_sequences`, thinking budgets, ...) straight to the SDK
  call. `AnthropicAdapter` supports a `cache_system_prompt` convenience flag
  for Anthropic prompt caching on that now-stable system block.
- Feature-flagged exact request/response console logging
  (`VERDIKT_LOG_LLM_CALLS`, on by default).
- YAML configuration with `${ENV_VAR}` interpolation, or pure-code
  configuration.
- `Verdikt.evaluate` / `evaluate_sync` / `evaluate_batch` facade.

[Unreleased]: https://github.com/jatin-mishra/verdikt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jatin-mishra/verdikt/releases/tag/v0.1.0
