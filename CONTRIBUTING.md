# Contributing to verdikt

Thanks for considering a contribution. This is a young project — small,
well-scoped PRs are the easiest to review and merge.

## Development setup

```bash
git clone https://github.com/jatin-mishra/verdikt.git
cd verdikt
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Running a judge for real (as opposed to the test suite, which uses a fake
LLM client and needs no keys) requires provider API keys:

```bash
export ANTHROPIC_AGENT_API_KEY=...   ANTHROPIC_AGENT_MODEL=claude-haiku
export GEMINI_API_KEY=...            GEMINI_AGENT_MODEL=gemini-2.5-flash
python examples/quickstart.py        # 41 runnable reference cases
```

## Before opening a PR

```bash
ruff check .              # lint — must be clean
python -m pytest tests -q # 65 tests, no network/API keys needed
```

If you change public behavior, also update:
- `README.md` (the relevant cookbook section, or the compatibility table)
- `CHANGELOG.md` under `[Unreleased]`
- `examples/verdikt.example.yaml` if it's a new judge type/execution mode
  (it's validated by `tests/test_example_yaml.py`)

## Project structure

| Path | What lives there |
|---|---|
| `verdikt/core/` | `EvalInput`/`Verdict`/config schemas, `BaseJudge`, the judge type registry |
| `verdikt/judges/` | Built-in judge types (pointwise, pairwise, rubric, ...) |
| `verdikt/llm/` | `LLMClient` interface, `FrontierClient` + provider SDK adapters, retry/cache wrappers |
| `verdikt/execution/` | Broadcast/judge-of-judges execution modes, consensus strategies |
| `verdikt/pipeline/` | Multi-step pipeline runner, aggregation, batch runner |
| `verdikt/prompts/templates/` | Jinja2 prompt templates, one per judge type |
| `tests/` | Uses `FakeLLMClient` (see `tests/conftest.py`) — no network calls |

See the README's [Extending verdikt](README.md#extending-verdikt) section
for the four supported extension points (custom judge, custom `LLMClient`,
custom protocol adapter, custom template directory) before adding something
new to the core package — most new capabilities should be a plugin, not a
core change.

## Reporting bugs / requesting features

Use the GitHub issue templates. For security vulnerabilities, see
[`SECURITY.md`](SECURITY.md) instead of opening a public issue.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` and add a dated entry to
   `CHANGELOG.md` (move `[Unreleased]` items under it).
2. Commit, tag `vX.Y.Z`, push the tag.
3. Draft a GitHub Release from that tag and publish it — this triggers
   `.github/workflows/publish.yml`, which builds and uploads to PyPI.

### One-time PyPI setup (before the first release)

`publish.yml` uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) instead of an API token secret. Once, on pypi.org:

1. Create the `verdikt` project on PyPI (or let the first trusted-publish
   run create it, if the name is available).
2. Under the project's **Publishing** settings (or, for a brand-new
   project, [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/)),
   add a trusted publisher:
   - Owner: `jatin-mishra`
   - Repository: `verdikt`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In the GitHub repo settings → Environments, create an environment named
   `pypi` (matches `environment.name` in `publish.yml`). No secrets needed
   there — the token exchange happens via OIDC at publish time.

After that one-time setup, every published GitHub Release publishes to
PyPI automatically.
