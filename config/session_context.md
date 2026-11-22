---
schema_version: 2
name: my-session
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    configuration: {}
    required:
      - lisa-pack-markdown
      - lisa-pack-personas
    # - lisa-pack-tavily
# Web search requires a web search provider pack, e.g., lisa-pack-tavily.
# Add one above then designate it as the web_search tool below.
# Then remove the hint from the bottom of the text below that.
providers:
  capabilities:
    memory: lisa-pack-memory
# tools: { web_search: lisa-pack-tavily }
routing: { from: { pack: lisa-pack-core, session: default } }
execution:
  max_model_calls: 24
  max_capability_calls: 24
  max_total_tokens: 250000
  max_completion_tokens: 24000
  max_wall_ms: 300000
  max_session_tokens: 5000000
  scopes:
    "lisa-pack-personas:researcher":
      max_model_calls: 8
      max_capability_calls: 8
      max_total_tokens: 120000
      max_completion_tokens: 12000
      max_wall_ms: 180000
---
# My Session

The user is running you via the CLI host at github.com/Lisa-Harness/lisa-cli.
However they have not yet installed any web search provider packs (e.g., lisa-pack-tavily).
The user should be reminded to remove this text once they have installed a web search provider pack.
