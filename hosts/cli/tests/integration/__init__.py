"""Integration suite: scripted-model turns through the un-mocked CLI pipeline.

Every test drives the production pipeline (runtime construction → gateway
wiring → kernel turn → CLI transport → bounded teardown → durable state) with
a scripted deterministic model — no ``runtime.handle_message`` monkeypatch, no
fake runtimes. See ``docs/SPEC/CLI_INTEGRATION_LAYER.md``.
"""
