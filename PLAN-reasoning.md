# PLAN — Max thinking effort + full reasoning-trace preserve/resend (OpenRouter)

Branch: `legacy`. Target: `hijarki-*` (and `ra-1*`) pipelines.

## Goal
For Claude/Fable/Opus-class models: run agents at `effort: "max"`, capture the FULL
reasoning trace (`reasoning` + `reasoning_details`) from every agent, resend native
`reasoning_details` verbatim into downstream agents, and surface traces to the API caller.

## Key OpenRouter facts (from docs)
- Unified param: `reasoning: {effort: max|xhigh|high|medium|low|minimal|none}` or `{max_tokens}`.
- OpenAI Python SDK has no `**kwargs` catch-all and `reasoning_effort` enum lacks `"max"`,
  so reasoning MUST be sent via `extra_body={"reasoning": {...}}`.
- Anthropic budget: `budget = min(max_tokens * 0.95, 128000)`, min 1024.
  `max_tokens` MUST be strictly greater than the budget or the response returns
  `finish_reason: "length"` with empty `content` (still billed).
- Trace location: non-stream `choices[].message.reasoning` / `.reasoning_details`;
  stream `choices[].delta.reasoning_details`.
- `reasoning_details` entries: `reasoning.text` (signed), `reasoning.encrypted`, `reasoning.summary`.
  Order must be preserved exactly when echoing back.
- Anthropic defaults to summarized thinking (`thinking.display: "summarized"`); the signed
  blocks are what enable faithful resend.

## Decisions
- Relay: resend native `reasoning_details` verbatim (order preserved) + keep text prefix.
- Effort config: per-agent `reasoning` → profile `reasoning` map → `default_reasoning` → env.
- max_tokens safety: auto-raise to `HIJARKI_REASONING_MAX_TOKENS` (default 32000) for max/xhigh on Claude.
- Detection: catalog-driven (gateway then OpenRouter public), TTL cache, minimal offline fallback.
- Invariant: user messages stay VERBATIM.

## Detection precedence (high → low)
1. Explicit profile/agent `reasoning` (unconditional).
2. Resolve via `MODEL_ALIASES`/`REVERSE_MODEL_ALIASES`.
3. Catalog match: `id`/`canonical_slug`/`alias_target.slug`, `architecture.tokenizer=="Claude"`,
   `instruct_type=="claude"`, author `anthropic`, or `CLAUDE_MODELS` env override.
4. Offline fallback hints (`anthropic/` prefix, `claude`, `fable`, `opus`, `sonnet`, `haiku`) + warning.

## Changes
- `main.py`:
  - catalog helpers: `_ensure_catalog`, `_harvest_catalog`, `_is_claude_model`,
    `_normalize_reasoning`, `_apply_reasoning`, `build_generation_config`.
  - `call_openrouter_agent`: `reasoning_config` param → `extra_body`; auto-raise max_tokens;
    capture `reasoning`/`reasoning_details`/`tool_calls`/`finish_reason`/`reasoning_tokens`.
  - `call_openrouter_agent_stream`: all generation params via `extra_body`; log reasoning.
  - `hijarki_caller`: forward `reasoning_config`.
  - `OpenAIResponseMessage`: nullable `content` + `reasoning`, `reasoning_details`, `tool_calls`;
    `OpenAIChoice.finish_reason` real value; surface non-stream + streaming delta.

## Pass-through fidelity fixes (same pass)
- `ChatCompletionRequest`: `extra="allow"` so ALL OpenAI params survive (were dropped).
- `build_generation_config(body)`: forwards every generation param via `extra_body`
  (SDK has no `**kwargs`); applied to hijarki, ra-1*, and direct paths.
- Direct passthrough (stream + non-stream): pass `chat_request.messages` verbatim
  (no system collapsing, no image/audio/file part dropping, no reordering).
- Response: `content` nullable (tool calls no longer coerced to `""`), `tool_calls` and real
  `finish_reason` preserved instead of hardcoded `"stop"`.
- Mirrored in `mainer.py` (legacy).
- `hijarki.py`:
  - `_resolve_reasoning`; pass to caller; keep reasoning on buffer results.
  - `_assistant_message` attaches native `reasoning_details` (else `reasoning`).
  - `HijarkiResult`: `master_reasoning` / `master_reasoning_details`; notes include traces.
- `profiles/o.json` master `reasoning: {effort: max}`; optional blocks in `full.json`/`light.json`.
- `test_hijarki_dryrun.py`: assert reasoning relay + order.
- `AGENTS.md`: document config + env vars.
- `mainer.py`: legacy sync.

## Env vars
- `HIJARKI_REASONING_EFFORT` — global default effort.
- `HIJARKI_REASONING_MAX_TOKENS` — safe max_tokens default (default 32000).
- `MODEL_CATALOG_TTL` — catalog cache seconds (default 86400).
- `CLAUDE_MODELS` — comma-separated extra Claude ids (gateway/private names).

## Verify (offline only)
`python3 -c "import ast; ast.parse(open('main.py').read())"`; `python3 test_hijarki_dryrun.py`;
`python3 -c "import main"`. Live curl DEFERRED until OpenRouter top-up.
Commit + push `origin legacy`.
