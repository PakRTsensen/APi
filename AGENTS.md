# AGENTS.md

## What this repo is
OpenAI-compatible proxy (FastAPI) that orchestrates multiple agent calls. The endpoint's
only jobs are: (a) build the `system` prompt per agent, (b) sequence agent calls. All user
content (text, images, audio, files, any OpenAI-compatible part) is forwarded VERBATIM —
never filtered or modified. Legacy "models" `ra-1`, `ra-1-pro` are orchestration workflows,
not real models; new `hijarki-*` models are the same idea but config-driven.

## Setup
- Deps: `pip install -r req.txt` (note: filename is `req.txt`, not `requirements.txt`).
  Python 3.10+ required (`str | List` unions); dev env is 3.12.
- Env: copy `.env.example` → `.env` (gitignored — never commit). Required vars:
  - `PROXY_AUTH_KEY` / `OPENROUTER_API_KEY` — comma-separated; keys rotated via `itertools.cycle`
  - `ROUTER_MODEL_PALETTE` — JSON string; every palette model ID must be listed here
  - `ROUTER_MODEL`, `OPENROUTER_MODEL_NAME`
- `OPENAI_BASE_URL` (optional, default `http://localhost:11434/v1/`): the base URL for ALL
  model calls (incl. the router). Keys are OpenRouter-style, so nothing works unless this
  points at a reachable OpenAI-compatible endpoint. `.env.example` still documents the old
  local default.
- Optional: `MODEL_ALIASES` (`orig:alias,...`) — resolved in both directions.

## Run / verify
- Start: `python main.py --rpm N` (0.0.0.0:8000). `--rpm` = per-request agent-call rate
  limit (0 = unlimited). Alternative: `uvicorn main:app`.
- No test suite. Focused verification:
  - `python3 -c "import ast; ast.parse(open('main.py').read()); print('ok')"`
  - `python3 test_hijarki_dryrun.py` — offline engine test (fake caller, no network);
    proves dynamic Buffer Zone ordering, cumulative relay, pass-through, notes persistence.
  - `python3 -c "import main"` (from repo root, with `.env` present)
  - Live: run server, `curl -s localhost:8000/v1/models` (auth: `Authorization: Bearer <PROXY_AUTH_KEY>`).
- Logs: per-session files under `logs/<proxy_key>/<ts>_<session_id>.log` (`logs/` gitignored).

## Runtime gotchas
- `main.py` vs `mainer.py`: near-duplicates; differ ONLY in the `skeptic_critic` prompt
  (`main.py` complete, `mainer.py` truncated). Keep both in sync; `mainer.py` is legacy.
- Agent calls: 7 retries, then an error marker (pipeline continues, never dies).

## Hijarki architecture (config-driven, dynamic)
- Models `hijarki-<stub>` map to `profiles/<stub>.json` (e.g. `hijarki-full` → `profiles/full.json`).
  Profiles are auto-discovered at startup and registered in `/v1/models`.
- Profile JSON is the SINGLE source of truth for system prompts: `sub_agents[].system_prompt`,
  `staf1`, `master`, `staf2`. NO prompts hardcoded in Python. Empty/missing `system_prompt`
  is fine (system message skipped) — engines still run.
- Engine: `hijarki.py` → `run_hijarki(...)`. Pure stdlib; MUST NOT import main.py.
- Flow: sub-agents (DYNAMIC count — 1, 30, 100, anything) run as a cumulative relay
  (each sees the user question + all previous sub-agent responses) → `staf1` sees all
  sub-agents → `master` sees all + `staf1` + session notes → `staf2` (last/lowest) records
  everything. Final HTTP output = ONLY master's response text (markdown code fences stripped).
- Buffer Zone: an in-memory `OrderedDict` holding every agent result (any count); `staf1`,
  `master`, `staf2` read the whole buffer produced before their phase.
- Per-agent model selection (each agent may use a DIFFERENT model). Resolution order:
  entry `"model"` → profile `"models"` map keyed by agent name → profile `"default_model"`
  → `fallback_model` arg (`OPENROUTER_MODEL_NAME`). Blank/whitespace is treated as unset.
- Context added AFTER user messages as `assistant` messages (`[Respon dari agent "X"]` prefix);
  session notes/glossary as one appended `user` message. User messages are never rewritten.
- Session notes (glossary) persist per conversation to `sessions/<session_id>.jsonl`
  (`sessions/` gitignored); later turns load prior lines and inject into Master + Staf 2.
- Streaming `stream: true` is supported via SSE keep-alive: server opens the response
  immediately and emits `"Praxis still thinking..."` chunks every `HIJARKI_KEEPALIVE_INTERVAL`
  seconds (default 15), then the master's final content + `[DONE]` only after the Buffer Zone
  completes. Non-streaming returns the same content in one shot.
- Keep `main.py`'s relay wiring intact: `call_openrouter_agent` expects `List[ChatMessage]`,
  but the engine passes `list[dict]`; the `handle_hijarki` closure converts dicts to
  `ChatMessage` before calling. `user_question` is extracted from the user's last text (only
  for prompt activation/logging) — attachments are never dropped.

## Reasoning / thinking trace (OpenRouter)
- Config precedence per agent: entry `"reasoning"` (str `"max"` or object
  `{"effort","max_tokens","enabled","exclude"}`) → profile `"reasoning"[<name>]` →
  profile `"default_reasoning"` → env `HIJARKI_REASONING_EFFORT`.
- Sent via `extra_body={"reasoning": ...}` (the OpenAI SDK has no `**kwargs`; `reasoning_effort`
  cannot express `"max"`). Only Claude/Fable/Opus-class models are touched.
- max_tokens safety: for `effort` `max`/`xhigh` the request `max_tokens` is auto-raised to
  `HIJARKI_REASONING_MAX_TOKENS` (default 32000; must exceed the Anthropic 95% budget or the
  response returns `finish_reason:"length"` with empty content).
- Trace capture: `reasoning`, `reasoning_details` (order preserved) and
  `usage.completion_tokens_details.reasoning_tokens` are kept per agent.
- Relay: downstream agents receive the previous agent's native `reasoning_details` verbatim on
  the assistant context message (fallback `reasoning`), plus the `[Respon dari agent "X"]` text.
  User messages are still never modified.
- Final response exposes master's `reasoning`/`reasoning_details` (non-stream and stream delta).
- Detection is catalog-driven: `_ensure_catalog` fetches `GET /models` from `OPENAI_BASE_URL`
  first, then public OpenRouter (`?model_authors=anthropic`), cached for `MODEL_CATALOG_TTL`
  (default 86400s). Match uses id/canonical_slug/alias_target, tokenizer/instruct_type,
  author `anthropic`, or `CLAUDE_MODELS` (comma-separated) override. Offline fallback hints:
  `anthropic/`, `claude`, `fable`, `opus`, `sonnet`, `haiku`.

## Pass-through fidelity (MUST preserve)
The proxy's only job is to build the system prompt and sequence agents. Everything else is
forwarded VERBATIM. Do not reintroduce filtering:
- `ChatCompletionRequest` and `ChatMessage` use `extra="allow"`; `build_generation_config(body)`
  forwards ALL generation params (top_p, top_k, seed, stop, tools, tool_choice, response_format,
  penalties, reasoning, ...) via `extra_body` — the OpenAI SDK has no `**kwargs`, so params that
  are not explicit SDK arguments MUST go through `extra_body`.
- Direct passthrough paths pass `chat_request.messages` as-is (no system collapsing, no dropping
  image/audio/file parts, no reordering).
- Responses are rebuilt from the upstream message dump (`raw_message`) so `refusal`, `annotations`,
  `audio`, `function_call`, etc. survive. `content` (nullable) is never coerced to `""`;
  `tool_calls` and the real `finish_reason`/`native_finish_reason` are preserved.
- `usage` token counts are summed across agents; `prompt_tokens_details`/`completion_tokens_details`
  (cached_tokens, reasoning_tokens, ...) are merged too.

## Git workflow (required)
- Active branch: `legacy`. Do NOT commit to `stable`/`experimental` (`origin/HEAD → stable`).
- Remote: `origin` = https://github.com/PakRTsensen/APi.git
- Project rule: EVERY change committed AND pushed to `origin legacy` immediately after
  verification — never leave uncommitted work.
- Commit style from history: conventional (`feat:`, `fix:`, `refactor:`, `config:`,
  `revert:`, ...); occasional plain short lines.