"""Dry-run test untuk engine hijarki.py — tanpa network, fake caller.

Memverifikasi:
(a) Buffer Zone dinamis: 6 sub-agent + staf1 + master + staf2 = 9 entri, urutan benar.
(b) Relay akumulatif: sub-agent ke-n melihat respon semua sub-agent sebelumnya.
(c) Staf1 melihat semua sub-agent; Master melihat sub-agent + staf1; staf2 melihat semua + master.
(d) Pesan user TIDAK dimodifikasi (pass-through verbatim).
(e) final_content = teks Master yang dibersihkan dari code fence.
(f) Catatan sesi (notes JSONL) ditulis berisi semua respon buffer.
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hijarki  # noqa: E402


def make_profile(n_sub: int) -> dict:
    sub = [
        {"name": f"sa_{i}", "system_prompt": f"prompt sa_{i}"}
        for i in range(1, n_sub + 1)
    ]
    return {
        "name": "test-dynamic",
        "default_model": "model-test",
        "models": {"staf_1": "model-map-staf1"},
        "sub_agents": sub,
        "staf1": {"name": "staf_1", "system_prompt": "prompt staf1"},
        "master": {"name": "master", "system_prompt": "prompt master", "model": "model-master-A"},
        "staf2": {"name": "staf_2", "system_prompt": "prompt staf2"},
    }


class FakeCaller:
    """Merekam semua pemanggilan; mengembalikan balasan yang bisa diverifikasi."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["agent_name"]
        messages = kwargs["messages"]
        # Konteks = semua pesan assistant (selain system/user asli)
        context = [m["content"] for m in messages if m["role"] == "assistant"]
        response_text = (
            f"RESPON_{name}||konteks={len(context)}||"
            + "|".join(str(c)[:40] for c in context)
        )
        return {
            "agent": name,
            "status": "success",
            "response_text": response_text,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }


def _fake_logger():
    import logging

    lg = logging.getLogger("dryrun")
    lg.setLevel(logging.WARNING)
    return lg


async def main() -> None:
    profile = make_profile(6)
    user_messages = [
        {"role": "user", "content": "pertanyaan asli"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:img"}}, {"type": "text", "text": "foto"}]},
    ]
    user_messages_snapshot = json.loads(json.dumps(user_messages))  # deep copy utk banding

    tmpdir = tempfile.mkdtemp(prefix="hijarki_test_")
    notes_path = os.path.join(tmpdir, "sesi.jsonl")

    caller = FakeCaller()
    result = await hijarki.run_hijarki(
        profile=profile,
        user_question="pertanyaan asli",
        user_messages=user_messages,
        session_logger=_fake_logger(),
        generation_config={"temperature": 0.7},
        caller=caller,
        fallback_model="fallback-model",
        notes_path=notes_path,
    )

    # (a) Buffer Zone berisi 9 entri, urutan: sa_1..sa_6, staf_1, master, staf_2
    names = list(result.buffer.keys())
    expected = ["sa_1", "sa_2", "sa_3", "sa_4", "sa_5", "sa_6", "staf_1", "master", "staf_2"]
    assert names == expected, f"urutan buffer salah: {names}"
    print(f"(a) buffer order ok: {len(names)} entri")

    # (c) Banyaknya konteks assistant per fase
    context_counts = [c["agent_name"] for c in caller.calls]
    assert context_counts == expected, context_counts
    # call 1 (sa_1): 0 konteks; call 2 (sa_2): 1; ...; call 6 (sa_6): 5
    for i, call in enumerate(caller.calls[:6]):
        n_assistant = sum(1 for m in call["messages"] if m["role"] == "assistant")
        assert n_assistant == i, f"sa_{i+1} harus melihat {i} konteks, dapat {n_assistant}"
    # staf1 (index 6): harus melihat 6 sub-agent
    assert sum(1 for m in caller.calls[6]["messages"] if m["role"] == "assistant") == 6
    # master (index 7): sub-agent (6) + staf1 (1) = 7
    assert sum(1 for m in caller.calls[7]["messages"] if m["role"] == "assistant") == 7
    # staf2 (index 8): sub-agent (6) + staf1 (1) + master (1) = 8
    assert sum(1 for m in caller.calls[8]["messages"] if m["role"] == "assistant") == 8
    print("(c) relay akumulatif + fase ok")

    # (d) Pesan user tidak dimodifikasi (pass-through verbatim) di SEMUA pemanggilan
    for call in caller.calls:
        sent = [dict(m) for m in call["messages"]]
        orig = [dict(m) for m in user_messages_snapshot]
        # Kedua pesan user asli harus muncul persis sama (role+content identik)
        found = [m for m in sent if any(m.get("role") == o.get("role") and m.get("content") == o.get("content") for o in orig)]
        assert len(found) == 2, f"user messages berubah di {call['agent_name']}: {sent}"
    print("(d) pass-through user messages verbatim ok")

    # (e) final_content: master menulis dengan code fence -> harus dibersihkan
    # Balasan master mentah (kalau ditulis dengan ```json ... ```), final_content harus
    # sama persis dengan isi dalam fence, tanpa fence.
    master_call = caller.calls[7]
    raw_master = master_call["messages"][-1]  # placeholder; engine pakai response_text
    # Ambil response_text asli yang dikembalikan fake caller
    raw = next(res for res in result.buffer.values() if res["agent"] == "master")["response_text"]
    fenced = "```json\n" + raw + "\n```"
    cleaned = hijarki._clean_final_content(fenced)
    assert cleaned == raw.strip(), (cleaned[:80], raw[:80])
    assert result.final_content == raw.strip(), result.final_content[:80]
    assert "```" not in result.final_content
    print("(e) final_content = teks Master bersih ok")

    # (f) notes file terpersistensi dengan semua respon buffer
    assert os.path.isfile(notes_path), "notes path tidak dibuat"
    with open(notes_path, "r", encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    assert len(lines) == 1, lines
    assert len(lines[0]["buffer"]) == 9, lines[0]["buffer"].keys()
    assert "RESPON_master" in lines[0]["master"]
    print("(f) persistensi catatan sesi ok")

    # Usage akumulasi
    assert result.prompt_tokens == 90 and result.completion_tokens == 180, (result.prompt_tokens, result.completion_tokens)
    print("(g) token usage terakumulasi ok")

    # (h) Resolusi model per-agent: prioritas model entri > models[nama] > default_model
    models_used = {c["agent_name"]: c["model_name"] for c in caller.calls}
    assert models_used["sa_1"] == "model-test", models_used["sa_1"]           # default_model
    assert models_used["staf_1"] == "model-map-staf1", models_used["staf_1"]  # models map
    assert models_used["master"] == "model-master-A", models_used["master"]   # model entri
    assert models_used["staf_2"] == "model-test", models_used["staf_2"]       # default_model
    print("(h) resolusi model per-agent ok")

    # (i) Profil "master saja" (tanpa sub_agents) valid, terdeteksi, dan berjalan
    await _test_master_only_profile()

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nSEMUA PENGUJIAN LULUS ✔")


async def _test_master_only_profile() -> None:
    """Regression: profil tanpa sub_agents (mis. hanya master) tidak boleh ditolak."""
    profile = {
        "name": "o",
        "master": {"name": "master", "system_prompt": "", "model": "model-only"},
    }
    assert hijarki._is_valid_profile(profile) is True
    assert hijarki._is_valid_profile({"sub_agents": []}) is False
    assert hijarki._is_valid_profile({"sub_agents": [], "staf2": {"name": "s2"}}) is True

    # discover + load dari file master-saja
    tmp = tempfile.mkdtemp(prefix="hijarki_o_")
    with open(os.path.join(tmp, "o.json"), "w", encoding="utf-8") as fh:
        json.dump(profile, fh)
    assert "o" in hijarki.discover_profiles(tmp), hijarki.discover_profiles(tmp)
    loaded = hijarki.load_profile("o", tmp)
    assert loaded is not None

    seen = []

    class Fake:
        async def __call__(self, **kw):
            seen.append((kw["agent_name"], kw["model_name"]))
            return {"agent": kw["agent_name"], "status": "success", "response_text": "M",
                    "prompt_tokens": 0, "completion_tokens": 0}

    res = await hijarki.run_hijarki(
        profile=loaded, user_question="q", user_messages=[{"role": "user", "content": "q"}],
        session_logger=_fake_logger(), generation_config={}, caller=Fake(), fallback_model="fb")
    assert seen == [("master", "model-only")], seen
    assert res.final_content == "M"
    shutil.rmtree(tmp, ignore_errors=True)
    print("(i) profil master-saja ok")


if __name__ == "__main__":
    asyncio.run(main())