"""Hijarki Praxis — engine orkestrasi multi-agent dinamis berbasis profil JSON.

Prinsip:
- Zero hardcoded prompt. Semua `system_prompt` dibaca VERBATIM dari file profil
  (profiles/*.json). Jika kosong/missing, pesan system dilewati (tidak error).
- Pass-through: pesan user (teks, gambar, file, part apa pun) diteruskan VERBATIM,
  tidak pernah dimodifikasi/difilter. Engine hanya (a) menambah pesan system dari
  profil, (b) MENAMBAHKAN message konteks SETELAH pesan user (respon agent
  sebelumnya sebagai role "assistant", plus catatan sesi/glosarium).
- Buffer Zone: OrderedDict dinamis yang menampung SEMUA respon agent (berapa pun
  jumlah sub-agent: 1, 30, 100, dst). Staf 1, Master, dan Staf 2 membaca seluruh
  isi buffer yang sudah ada saat fase mereka berjalan.
- Fase: sub-agents (relay akumulatif) -> staf1 -> master -> staf2 (pencatat, paling
  bawah). Staf2 mencatat dinamika/glosarium ke file JSONL persisten per sesi.
- Output final: HANYA teks respon Master (dibersihkan dari code fence markdown),
  via atribut `final_content`.

Murni stdlib, tidak mengimpor main.py (hindari circular import).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from typing import Callable, List, Optional

logger = logging.getLogger("hijarki")

DEFAULT_PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")

# Pola code fence markdown di awal/akhir keluaran yang tidak diinginkan.
_LEADING_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n", re.MULTILINE)
_TRAILING_FENCE = re.compile(r"\n?\s*```\s*$")


class HijarkiResult:
    """Hasil satu putaran pipeline Hijarki."""

    __slots__ = ("final_content", "buffer", "prompt_tokens", "completion_tokens")

    def __init__(
        self,
        final_content: str,
        buffer: "OrderedDict[str, dict]",
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.final_content = final_content
        self.buffer = buffer
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _profiles_dir(profiles_dir: Optional[str]) -> str:
    return profiles_dir or DEFAULT_PROFILES_DIR


def _is_valid_profile(data: object) -> bool:
    """Profil valid bila punya minimal satu agent yang bisa dijalankan.

    Agent dianggap ada bila `sub_agents` berisi minimal satu entri, ATAU salah
    satu peran staf/master (`staf1`, `master`, `staf2`) ada. Ini mengizinkan
    profil "master saja" (tanpa sub-agent) tetap terdaftar dan dapat dipanggil.
    """
    if not isinstance(data, dict):
        return False
    subs = data.get("sub_agents")
    if isinstance(subs, list) and len(subs) > 0:
        return True
    for role in ("staf1", "master", "staf2"):
        if isinstance(data.get(role), dict):
            return True
    return False


def discover_profiles(profiles_dir: Optional[str] = None) -> List[str]:
    """Mengembalikan daftar nama profil (stub, tanpa .json) yang valid di folder profil.

    Sebuah file dianggap profil valid jika parse JSON-nya sukses dan memiliki
    minimal satu agent (sub-agent atau staf/master). Hasil diurutkan alfabetis.
    """
    directory = _profiles_dir(profiles_dir)
    found: List[str] = []
    if not os.path.isdir(directory):
        return found
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(".json"):
            continue
        stub = entry[: -len(".json")]
        try:
            with open(os.path.join(directory, entry), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - profil rusak tidak boleh membunuh startup
            logger.warning("Profil %s gagal dibaca: %s", entry, exc)
            continue
        if _is_valid_profile(data):
            found.append(stub)
    return found


def load_profile(profile_name: str, profiles_dir: Optional[str] = None) -> Optional[dict]:
    """Memuat profil JSON berdasarkan nama stub. Mengembalikan dict atau None."""

    directory = _profiles_dir(profiles_dir)
    path = os.path.join(directory, f"{profile_name}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Profil %s tidak dapat dimuat: %s", profile_name, exc)
        return None
    if not _is_valid_profile(data):
        logger.warning("Profil %s tidak memiliki agent (sub_agents/staf1/master/staf2).", profile_name)
        return None
    return data


def _load_notes(notes_path: Optional[str]) -> List[str]:
    """Membaca baris JSONL catatan sesi lama (glosarium persisten)."""

    if not notes_path or not os.path.isfile(notes_path):
        return []
    lines: List[str] = []
    try:
        with open(notes_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    lines.append(line)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal membaca catatan sesi %s: %s", notes_path, exc)
    return lines


def _append_note(notes_path: Optional[str], record: dict) -> None:
    """Menambahkan satu rekaman JSONL catatan sesi (append-only, persisten)."""

    if not notes_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(notes_path)), exist_ok=True)
        with open(notes_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gagal menulis catatan sesi %s: %s", notes_path, exc)


def _clean_final_content(text: str) -> str:
    """Menghapus code fence markdown di awal/akhir; konten dalam tetap utuh."""

    cleaned = _LEADING_FENCE.sub("", text, count=1)
    cleaned = _TRAILING_FENCE.sub("", cleaned)
    return cleaned.strip()


def _system_message(system_prompt: str) -> Optional[dict]:
    """Pesan system hanya jika system_prompt tidak kosong (verbatim)."""

    if not system_prompt or not system_prompt.strip():
        return None
    return {"role": "system", "content": system_prompt}


def _assistant_message(name: str, response_text: str) -> dict:
    return {"role": "assistant", "content": f'[Respon dari agent "{name}"]\n{response_text}'}


def _notes_message(lines: List[str]) -> dict:
    joined = "\n".join(lines) if lines else "(belum ada catatan sesi)"
    return {"role": "user", "content": "Catatan sesi (glosarium):\n" + joined}


async def run_hijarki(
    *,
    profile: dict,
    user_question: str,
    user_messages: List[dict],
    session_logger: logging.Logger,
    generation_config: dict,
    caller: Callable,
    fallback_model: str,
    notes_path: Optional[str] = None,
) -> HijarkiResult:
    """Menjalankan pipeline Hijarki sesuai profil.

    Args:
        profile: dict profil (hasil `load_profile`).
        user_question: teks pertanyaan user (untuk log/rekaman, TIDAK menggantikan
            konten user).
        user_messages: daftar dict pesan user VERBATIM (tidak pernah dimodifikasi).
        session_logger: logger per sesi.
        generation_config: config generasi ({temperature, max_tokens?}).
        caller: coroutine async panggil model:
            await caller(agent_name=..., model_name=..., system_prompt=...,
                         messages=[...dicts...], generation_config=...)
            -> dict {agent, status, response_text, prompt_tokens, completion_tokens,
                     total_tokens} (format hasil `call_openrouter_agent` di main.py).
        fallback_model: model default bila profil/agent tidak menentukan.
        notes_path: path file JSONL catatan sesi (persisten). None = tanpa persistensi.
    """

    session_logger.info("--- Hijarki pipeline mulai (profil: %s) ---", profile.get("name", "?"))
    session_logger.info("Jumlah sub-agent: %d", len(profile.get("sub_agents", [])))

    # --- Susun urutan agent: sub-agents -> staf1 -> master -> staf2 ---
    entries: List[dict] = []
    for sa in profile.get("sub_agents", []):
        entries.append({"kind": "sub_agent", **sa})
    for key in ("staf1", "master", "staf2"):
        entry = profile.get(key)
        if isinstance(entry, dict):
            entries.append({"kind": key, **entry})

    # --- Buffer Zone: dinamis, menampung SEMUA respon sesuai urutan eksekusi ---
    buffer: "OrderedDict[str, dict]" = OrderedDict()

    notes = _load_notes(notes_path)

    profile_models = profile.get("models") if isinstance(profile.get("models"), dict) else {}

    def _resolve_model(entry: dict, name: str) -> str:
        """Prioritas model per agent:
        1. `model` di entri agent itu sendiri
        2. `models[<name>]` di level profil (map nama agent -> model)
        3. `default_model` profil
        4. `fallback_model` yang diberikan pemanggil
        Nilai kosong/whitespace dianggap tidak diset.
        """
        candidates = (
            entry.get("model"),
            profile_models.get(name),
            profile.get("default_model"),
            fallback_model,
        )
        for cand in candidates:
            if isinstance(cand, str) and cand.strip():
                return cand.strip()
        return fallback_model

    for idx, entry in enumerate(entries):
        name = entry.get("name") or entry.get("kind") or f"agent_{idx}"
        system_prompt = entry.get("system_prompt") or ""
        model_name = _resolve_model(entry, name)

        # (1) system message dari profil (verbatim), (2) pesan user VERBATIM,
        # (3) konteks tambahan: respon agent yang sudah ada di buffer + catatan sesi.
        messages: List[dict] = []
        sys_msg = _system_message(system_prompt)
        if sys_msg is not None:
            messages.append(sys_msg)
        # Salinan dangkal — dict pesan user tidak pernah diubah oleh engine.
        messages.extend(list(user_messages))
        for prev_name, prev_result in buffer.items():
            prev_text = prev_result.get("response_text") or ""
            messages.append(_assistant_message(prev_name, prev_text))
        if entry["kind"] in ("master", "staf2") and notes:
            messages.append(_notes_message(notes))

        session_logger.info(
            "Agent '%s' (fase %s, model %s): buffer size %d",
            name, entry["kind"], model_name, len(buffer),
        )

        result = await caller(
            agent_name=name,
            model_name=model_name,
            system_prompt=system_prompt,
            messages=messages,
            generation_config=generation_config,
        )
        if not isinstance(result, dict):
            result = {"agent": name, "status": "error", "response_text": "", "prompt_tokens": 0, "completion_tokens": 0}
        if result.get("status") != "success":
            # Jangan hentikan pipeline; tandai di buffer agar agent lain tahu.
            result["response_text"] = f"[Agent error: {result.get('error', 'unknown')}]"
            session_logger.warning("Agent '%s' gagal; pipeline tetap lanjut.", name)
        buffer[name] = result
        session_logger.info("Buffer Zone size: %d", len(buffer))

    # --- Tentukan output final: respon Master (atau hasil terakhir bila tak ada master) ---
    master_result = buffer.get("master")
    if master_result is not None:
        final_content = _clean_final_content(master_result.get("response_text") or "")
    elif buffer:
        final_content = _clean_final_content(next(reversed(buffer.values())).get("response_text") or "")
    else:
        final_content = ""

    # --- Persistensi catatan sesi (Staf 2 / Master) ---
    if notes_path:
        record: dict = {
            "ts": __import__("time").time(),
            "user_question": user_question,
            "master": (master_result or {}).get("response_text", ""),
            "staf2": (buffer.get("staf2") or {}).get("response_text", ""),
            "buffer": {bname: bres.get("response_text", "") for bname, bres in buffer.items()},
        }
        _append_note(notes_path, record)

    prompt_tokens = sum(int(res.get("prompt_tokens", 0) or 0) for res in buffer.values())
    completion_tokens = sum(int(res.get("completion_tokens", 0) or 0) for res in buffer.values())

    session_logger.info("--- Hijarki pipeline selesai (profil: %s) ---", profile.get("name", "?"))
    return HijarkiResult(
        final_content=final_content,
        buffer=buffer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )