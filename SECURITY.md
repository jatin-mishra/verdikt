# Security Policy

## Supported Versions

verdikt is pre-1.0; only the latest published release on PyPI and the `main`
branch are supported with security fixes.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report it privately via one of:

- [GitHub Security Advisories](https://github.com/jatin-mishra/verdikt/security/advisories/new)
  for this repository (preferred — lets us coordinate a fix before disclosure)
- Email **jatinm1shra10cr@gmail.com** with a description of the issue, steps
  to reproduce, and its potential impact

You should expect an initial response within a few days. Once a fix is
available, we'll credit you in the release notes (unless you'd prefer to stay
anonymous) and coordinate a disclosure timeline with you.

## Scope

verdikt itself does not run a server and has no network-facing surface of its
own; it makes outbound calls to the LLM providers you configure. Relevant
security concerns for this project include (non-exhaustively):

- Prompt-injection-resistant handling of untrusted content in judge templates
  (see `verdikt/prompts/templates/_macros.j2`'s `untrusted()`/`security_note()`
  macros)
- Handling of API keys/credentials in `ProviderConfig` and the YAML config
  loader
- Dependency vulnerabilities in `anthropic`, `google-genai`, `httpx`,
  `jinja2`, `pydantic`, or `pyyaml`
