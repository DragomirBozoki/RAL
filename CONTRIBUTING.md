# Contributing to RAL

Thanks for helping. RAL is early, so please open an issue before large changes.

## Clean-room rule

RAL is written from scratch from public documentation, papers and permissively licensed
libraries.

- Do not copy, adapt or reconstruct code from an employer's or a client's codebase, or from
  any source whose licence does not allow it.
- Cite public references (papers, official documentation) in docstrings for non-trivial
  algorithms.
- If you are unsure whether something may be contributed, ask in an issue first.

## Licences of dependencies

The core is Apache-2.0. Core runtime dependencies must be permissive (MIT, BSD, Apache-2.0,
ISC, PSF). GPL, AGPL and LGPL dependencies are not accepted in the core. CI checks this.
Vendor-specific code (robot message definitions, decoders) belongs in separate plugin
packages, never in the core.

## Developer Certificate of Origin (DCO)

Every commit must be signed off, which certifies the
[Developer Certificate of Origin 1.1](https://developercertificate.org/):

```bash
git commit -s -m "feat(core): add ring buffer"
```

This adds a `Signed-off-by: Your Name <you@example.com>` line with your real name.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat`, `fix`, `docs`,
`test`, `refactor`, `build`, `ci`, `chore`, optionally with a scope such as `feat(core): ...`.

## Development setup

```bash
uv sync --all-extras
uv run pre-commit install
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest                 # unit and integration tests, no hardware needed
uv run pytest -m hardware     # needs real audio devices
```

Tests must not need hardware. Use `FileBackend`, synthetic signals and fakes; tests that need
real devices carry `@pytest.mark.hardware`.

## Privacy and secrets

Never commit keys, tokens, real deployment addresses, customer names or recordings of people.
Use `.env` (ignored) with `.env.example` as the template.
