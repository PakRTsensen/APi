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
- Context added AFTER user messages as `assistant` messages (`[Respon dari agent "X"]` prefix);
  session notes/glossary as one appended `user` message. User messages are never rewritten.
- Session notes (glossary) persist per conversation to `sessions/<session_id>.jsonl`
  (`sessions/` gitignored); later turns load prior lines and inject into Master + Staf 2.
- Non-streaming only: `stream: true` for a `hijarki-*` model → 400.
- Keep `main.py`'s relay wiring intact: `call_openrouter_agent` expects `List[ChatMessage]`,
  but the engine passes `list[dict]`; the `handle_hijarki` closure converts dicts to
  `ChatMessage` before calling. `user_question` is extracted from the user's last text (only
  for prompt activation/logging) — attachments are never dropped.

## Git workflow (required)
- Active branch: `legacy`. Do NOT commit to `stable`/`experimental` (`origin/HEAD → stable`).
- Remote: `origin` = https://github.com/PakRTsensen/APi.git
- Project rule: EVERY change committed AND pushed to `origin legacy` immediately after
  verification — never leave uncommitted work.
- Commit style from history: conventional (`feat:`, `fix:`, `refactor:`, `config:`,
  `revert:`, ...); occasional plain short lines.