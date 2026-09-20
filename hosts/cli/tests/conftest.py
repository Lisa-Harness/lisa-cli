"""Shared fixtures for the ``hosts/cli`` test suite.

Hermetic environment contract
-----------------------------
Every test in this suite runs against a **scrubbed** environment: the
autouse ``_hermetic_env`` fixture removes all Lisa/provider configuration
variables from ``os.environ`` before each test, so a test sees the
framework defaults unless it sets a variable itself.

Why: the CLI host loads an operator ``.env`` at *module import time*
(``hosts.cli.main`` runs ``load_dotenv`` during collection, searching
``Path.cwd()/.env`` first), so ambient operator configuration — provider
keys, ``LISA_DATA_DIR``, a stale ``LISA_SESSION_CONTEXT_PATH`` — lands in
``os.environ`` before any test code runs. Tests then failed on the
operator's machine but passed on CI (the 2026-09-19 once-decisions
incident: 22/22 red because ``.env`` pointed the session context at a
deleted file). A dedicated test ``.env`` cannot fix this — the operator
file wins the search order, and the load precedes any fixture — so the
suite instead declares that tests own their environment.

Tests that need a variable set it explicitly (``monkeypatch.setenv``);
tests that need to observe the "no keys" path rely on the scrub rather
than enumerating vars themselves. ``live``-marked tests may set real
keys inside the test body; the scrub runs first, the test's own values
win afterward.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

#: Variable-name prefixes scrubbed before every test. Lisa configuration,
#: model-provider credentials, and the CLI's tool-adjacent knobs. Anything
#: NOT listed survives (PATH, HOME, PYTHON*, UV*, VIRTUAL_ENV, LANG, ...) —
#: the scrub removes Lisa *config*, never interpreter/tooling state.
_SCRUB_PREFIXES: tuple[str, ...] = (
    # Lisa framework configuration (host/config.py, sdk/*, host/paths.py).
    "LISA_",
    # Model-provider credentials and endpoints (gateway_provider, probes).
    "OPENAI_",
    "AZURE_",
    "ANTHROPIC_",
    "GOOGLE_",
    "GEMINI_",
    "OLLAMA_",
    "OPENROUTER_",
    "AWS_",
    # Pack-provided service credentials observed in operator .env files.
    "BRAVE_",
    "TAVILY_",
    "SERPER_",
    "CLOUDFLARE_",
    "CLOUDSERVERS_",
    "GITHUB_",
    # Framework-tunable knobs exported alongside the credentials.
    "STAGE_",
    "DISTILL_",
    # SDK content/venv discovery (pack discovery must be test-controlled).
    "SDK_",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove ambient Lisa/provider config vars for the duration of a test.

    Uses ``monkeypatch.delenv`` so pytest restores the original environ
    on teardown — the scrub is per-test, never a process-wide mutation.
    Variables set by the test body itself (before or after this fixture
    runs) are unaffected: this only scrubs what was ambient at entry.
    """
    for name in list(os.environ):
        if name.startswith(_SCRUB_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    yield