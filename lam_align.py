"""lam_align.py — adattatore per il backend del sito (main.py).

Espone `align_as_scribe_data(vocals, text, lam_dir, synced_lyrics=...)` che main.py
chiama nel ramo "LAM in prima battuta". Delega all'allineatore LAM collaudato sul PC
(`lam_local.py`): allinea il TESTO dato al canto col modello LAM (checkpoint), applica
ancore synced + anti-deragliamento + check di copertura, e restituisce un dict
"scribe-like" ({ok, words:[{type,text,start,end,speaker_id}], text, reason, big_gaps}).

DEPLOY (accanto a main.py sul server), come da memoria:
  - questo file `lam_align.py`
  - `lam_local.py` (dal PC, whisperX-main)
  - il repo LAM in `SINGING_LAM_DIR` (default /opt/karaokecreator/lam) con
    `checkpoints/checkpoint_Baseline` + i .py del repo (model.py, utils.py, data.py…)
  - venv: torch/torchaudio (cpu), transformers, soundfile, phonemizer, librosa,
    sortedcontainers, tqdm, pandas, scipy  +  espeak-ng di sistema
Fail-safe: qualunque problema qui -> main.py cade su Scribe.
"""
import os

try:
    import lam_local
except Exception as _e:                     # import fallito -> main.py usera' Scribe
    lam_local = None
    _IMPORT_ERR = _e


def align_as_scribe_data(vocals_path, text, lam_dir=None, synced_lyrics=None,
                         ffmpeg="ffmpeg"):
    if lam_local is None:
        return {"ok": False, "words": [], "text": text or "",
                "reason": f"lam_local non importabile: {_IMPORT_ERR}"}
    if not text or not str(text).strip():
        return {"ok": False, "words": [], "text": "", "reason": "testo vuoto"}

    # il repo LAM (checkpoint) sta in lam_dir: punta lam_local lì (deve contenere
    # checkpoints/checkpoint_Baseline + i sorgenti del repo LAM).
    if lam_dir:
        lam_local.LAM_DIR = str(lam_dir)

    try:
        lrc, words, info = lam_local.align_to_lrc(
            str(vocals_path), str(text), synced_lyrics=synced_lyrics, ffmpeg=ffmpeg)
    except Exception as e:
        return {"ok": False, "words": [], "text": text or "",
                "reason": f"errore allineamento LAM: {e}"}

    scribe_words = []
    for w in (words or []):
        try:
            scribe_words.append({
                "type": "word",
                "text": w["word"],
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "speaker_id": "",
            })
        except Exception:
            continue

    info = info or {}
    ok = bool(info.get("ok") and scribe_words)
    return {
        "ok": ok,
        "words": scribe_words,
        "text": str(text),
        "language_code": "ita",
        "lrc": lrc or "",
        "reason": info.get("reason"),
        "big_gaps": info.get("big_gaps"),
        "anchored": info.get("anchored"),
    }
