"""
lam_local.py — Allineamento karaoke LOCALE con LAM (LyricsAlignment-Multilingual),
il modello contrastive per il CANTATO. Sostituisce completamente ElevenLabs Scribe:
nessuna API, nessun costo, gira sulla GPU del PC (o CPU se manca CUDA).

Serve il TESTO del brano (da LRCLIB o incollato): LAM ALLINEA, non trascrive.

API principale:
    align_to_lrc(vocals_path, text, synced_lyrics=None) -> (lrc_enhanced, words)

Requisiti gia' presenti sul PC: torch, soundfile, librosa, phonemizer + eSpeak NG.
"""
import os, sys, csv, re, unicodedata, threading, tempfile, shutil, subprocess

# --- Hosting condiviso (CloudLinux LVE): OpenBLAS/torch aprono 1 thread per CORE
# (qui 32) e saturano il limite processi (RLIMIT_NPROC ~1500) -> "pthread_create
# failed" -> SIGSEGV. Vanno forzati a 1 thread PRIMA di importare numpy/torch. ---
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# TMPDIR ESEGUIBILE: phonemizer copia la lib espeak in una tempdir e la carica con
# dlopen; se /tmp e' noexec -> "failed to map segment from shared object". Punto la
# tempdir a una cartella in home (eseguibile), PRIMA di importare tempfile/phonemizer.
_xt = os.path.join(os.path.expanduser("~"), ".karaoke_tmp")
try:
    os.makedirs(_xt, exist_ok=True)
    os.environ["TMPDIR"] = _xt
    os.environ["TMP"] = _xt
    os.environ["TEMP"] = _xt
    tempfile.tempdir = _xt     # forza: tempfile cacha la dir alla 1a chiamata
except Exception:
    pass

# --- eSpeak NG: phonemizer ha bisogno del path della libreria ---
if os.name == "nt" and not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
    for _p in (r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
               r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll"):
        if os.path.exists(_p):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _p
            break
elif os.name != "nt" and not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
    import glob as _glob
    _home = os.path.expanduser("~")
    _picked = None
    # 1) espeak-ng COMPILATO in home (~/espeak-ng-local): usa la glibc di sistema,
    # quindi carica dove i wheel manylinux falliscono ("GLIBC_2.29 not found").
    for _p in sorted(_glob.glob(os.path.join(_home, "espeak-ng-local", "lib", "libespeak-ng.so*")),
                     reverse=True):
        if os.path.exists(_p):
            _picked = _p
            _dd = os.path.join(_home, "espeak-ng-local", "share", "espeak-ng-data")
            if os.path.isdir(_dd):
                os.environ.setdefault("ESPEAK_DATA_PATH", _dd)
            break
    # 2) espeakng-loader (pip): comodo ma spesso glibc-incompatibile
    if not _picked:
        try:
            import espeakng_loader as _enl
            _picked = _enl.get_library_path()
            try:
                os.environ.setdefault("ESPEAK_DATA_PATH", _enl.get_data_path())
            except Exception:
                pass
        except Exception:
            pass
    # 3) fallback: .so di sistema/conda
    if not _picked:
        for _base in (os.path.join(_home, "miniconda3", "lib"),
                      "/usr/lib64", "/usr/lib", "/usr/local/lib"):
            _c = sorted(_glob.glob(os.path.join(_base, "libespeak-ng.so*")))
            if _c:
                _picked = _c[0]
                break
    if _picked:
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _picked

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
LAM_DIR = os.path.join(_HERE, "lam")
_RES = 256 / 22050 * 3
GAP_STRUM = 8.0

_LOCK = threading.Lock()
_MODEL = {}
_ESPEAK = {}


def _LOG(msg):
    """Log configurabile. Di default stampa (stdout). Il main.py del server lo
    sostituisce con log.info cosi' i messaggi [LAM] finiscono nel karaoke_api.log."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------- testo ----------
def _norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def clean_lines(text):
    """Righe cantabili: salta header (fino a 'BY ...'), WWW, separatori, vuote."""
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = 0
    for i, ln in enumerate(raw[:6]):
        if ln.strip().upper().startswith("BY "):
            start = i + 1
            break
    out = []
    for ln in raw[start:]:
        s = ln.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("WWW.") or up.startswith("BY ") or set(s) <= set("-=_ "):
            continue
        out.append(s)
    return out


# ---------- modello ----------
def _load_lam():
    with _LOCK:
        if "m" not in _MODEL:
            if LAM_DIR not in sys.path:
                sys.path.insert(0, LAM_DIR)
            import torch
            try:
                torch.set_num_threads(1)   # hosting LVE: niente esplosione di thread
            except Exception:
                pass
            from model import AcousticModel
            import utils
            dev = _device()
            model = AcousticModel(1, 256, 72, 32, 1, 0.1)
            utils.load_model(model, os.path.join(LAM_DIR, "checkpoints", "checkpoint_Baseline"), False)
            model = model.to(dev)
            model.eval()
            _MODEL["m"], _MODEL["utils"], _MODEL["dev"] = model, utils, dev
            print(f"[LAM] modello caricato su {dev.upper()}")
        return _MODEL["m"], _MODEL["utils"], _MODEL["dev"]


def _espeak_it():
    with _LOCK:
        if "b" not in _ESPEAK:
            # Aggancia ESPLICITAMENTE la libreria eSpeak NG di espeakng_loader:
            # la sola env PHONEMIZER_ESPEAK_LIBRARY spesso non basta (phonemizer
            # dice "espeak not installed"). EspeakWrapper.set_library e' il metodo
            # affidabile.
            try:
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
                _lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
                if not _lib:
                    try:
                        import espeakng_loader as _enl
                        _lib = _enl.get_library_path()
                    except Exception:
                        _lib = None
                if _lib:
                    EspeakWrapper.set_library(_lib)
            except Exception:
                pass
            try:
                from phonemizer.backend import EspeakBackend
                from phonemizer.separator import Separator
                from phonemizer.punctuation import Punctuation
            except ImportError as _ie:
                # Messaggio diagnostico: quasi sempre il programma gira con un
                # interprete Python DIVERSO da quello in cui e' stato installato
                # phonemizer. Qui si vede subito quale.
                raise ImportError(
                    f"{_ie}\n\nPython in uso: {sys.executable}\n"
                    f"Installa la dipendenza con:\n"
                    f'   "{sys.executable}" -m pip install phonemizer\n'
                    f"(serve anche eSpeak NG, gia' presente sul sistema)"
                ) from _ie
            _ESPEAK["b"] = EspeakBackend("it")
            _ESPEAK["sep"] = Separator(phone=';', word=None)
            _ESPEAK["punct"] = Punctuation(';:,.!"?()-')
        return _ESPEAK["b"], _ESPEAK["sep"], _ESPEAK["punct"]


def _write_annot(words, annot_dir, name="S"):
    b, sep, punct = _espeak_it()
    rows = []
    st = ed = 0
    for w in words:
        cleaned = punct.remove(w).lower()
        try:
            phone = b.phonemize([cleaned], separator=sep, strip=True)[0]
        except Exception:
            phone = ""
        if not phone:
            phone = "a"
        ed += len(phone.split(";"))
        rows.append([w, phone, [st, ed]])
        st = ed + 1
        ed = st
    with open(os.path.join(annot_dir, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["word", "phonemizer", "phone_idx"])
        wr.writerows(rows)


def _lam_word_times(audio_dir, annot_dir, sr=22050):
    import torch, torch.nn as nn, torch.nn.functional as F
    model, utils, dev = _load_lam()
    from data import JamendoLyricsDataset
    from model import train_audio_transforms
    ds = JamendoLyricsDataset(sr, audio_dir, annot_dir, ".wav")
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                     collate_fn=utils.my_collate)
    import time as _t
    with torch.no_grad():
        for _data in dl:
            x, idx, meta = _data
            idx = idx[0][0]
            words, audio_name, audio_length = meta[0]
            x = x.reshape(1, 1, -1)
            x = utils.move_data_to_device(x, dev)
            x = x.squeeze(0).squeeze(1)
            x = train_audio_transforms.to(dev)(x)
            x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1)
            _t0 = _t.time()
            out = F.log_softmax(model(x), dim=2)
            _, _, ncls = out.shape
            song_pred = out.data.cpu().numpy().reshape(-1, ncls)
            _LOG(f"[LAM]   forward modello: {_t.time()-_t0:.1f}s (frames={song_pred.shape[0]})")
            total = int(audio_length / sr // _RES)
            song_pred = song_pred[:total, :]
            song_pred = np.log(np.exp(song_pred) + np.random.uniform(1e-11, 1e-10, song_pred.shape))
            _t0 = _t.time()
            word_align, score = utils.alignment(song_pred, words, idx)
            _LOG(f"[LAM]   viterbi alignment: {_t.time()-_t0:.1f}s")
            return [(w[0] * _RES, w[1] * _RES) for w in word_align], float(score)
    return [], -1e9


# ---------- rifiniture ----------
def _note_onsets(wav_path):
    """Attacchi di NOTA (spectral flux): sul cantato l'attacco e' morbido e il
    modello marca la sillaba dopo l'inizio percettivo della nota."""
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        return np.asarray(librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True), dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _snap_to_note_onsets(starts, note_onsets, max_shift=0.35):
    if len(note_onsets) == 0:
        return starts, 0
    out = list(starts)
    moved = 0
    for i, t in enumerate(out):
        prev = note_onsets[note_onsets <= t]
        if len(prev) == 0:
            continue
        cand = float(prev[-1])
        if 0 < (t - cand) <= max_shift and (i == 0 or cand > out[i - 1] + 0.02):
            out[i] = cand
            moved += 1
    return out, moved


def _voice_onsets(wav_path):
    import soundfile as sf
    d, sr = sf.read(wav_path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(1)
    hop, win = int(0.01 * sr), int(0.02 * sr)
    n = max(0, (len(d) - win) // hop)
    rms = np.array([np.sqrt(np.mean(d[i * hop:i * hop + win] ** 2)) for i in range(n)])
    peak = np.percentile(rms, 99) or 1e-6
    thr = max(peak * 0.12, 0.02)
    voiced = rms > thr
    onsets = []
    for i in range(1, len(voiced) - 50):
        if voiced[i] and not voiced[i - 1] and np.mean(voiced[i:i + 50]) >= 0.55:
            t = i * 0.01
            if not onsets or t - onsets[-1] > 0.20:
                onsets.append(t)
    return np.array(onsets) if onsets else np.array([0.0])


def _voice_onsets_fine(wav_path):
    """Attacchi vocali FITTI (ogni ripresa di voce dopo un micro-silenzio): servono
    per agganciare le PAROLE al canto reale, non solo gli inizi-riga."""
    import soundfile as sf
    d, sr = sf.read(wav_path, dtype="float32")
    if d.ndim > 1:
        d = d.mean(1)
    hop, win = int(0.01 * sr), int(0.02 * sr)
    n = max(0, (len(d) - win) // hop)
    rms = np.array([np.sqrt(np.mean(d[i * hop:i * hop + win] ** 2)) for i in range(n)])
    peak = np.percentile(rms, 99) or 1e-6
    thr = max(peak * 0.10, 0.018)
    voiced = rms > thr
    onsets = []
    for i in range(1, len(voiced) - 6):
        # ripresa di voce sostenuta per almeno ~60ms dopo un attimo di calo
        if voiced[i] and not voiced[i - 1] and np.mean(voiced[i:i + 6]) >= 0.6:
            t = i * 0.01
            if not onsets or t - onsets[-1] > 0.12:
                onsets.append(t)
    return np.array(onsets) if onsets else np.array([0.0])


def parse_synced_lyrics(synced):
    """[mm:ss.xx] testo  ->  [(sec, testo)] (tempi umani da karaoke: ottime ancore)."""
    out = []
    for ln in str(synced or "").splitlines():
        m = re.match(r"\s*\[(\d+):(\d+(?:[.,]\d+)?)\]\s*(.*)", ln)
        if not m:
            continue
        sec = int(m.group(1)) * 60 + float(m.group(2).replace(",", "."))
        txt = (m.group(3) or "").strip()
        if txt:
            out.append((sec, txt))
    return out


def _anchor_lines_to_synced(lines, line_start, synced_lines, max_shift=6.0):
    if not synced_lines:
        return line_start, 0
    def key(s):
        s = _norm(s)
        return "".join(ch for ch in s if ch.isalnum() or ch == " ").strip()
    sk = [(t, key(txt)) for t, txt in synced_lines]
    used, anchored = -1, 0
    out = dict(line_start)
    for li, ln in enumerate(lines):
        k = key(ln)
        if not k:
            continue
        for j in range(used + 1, len(sk)):
            t, s = sk[j]
            if not s:
                continue
            if s == k or s.startswith(k[:18]) or k.startswith(s[:18]):
                if abs(t - out.get(li, t)) <= max_shift:
                    out[li] = t
                    used = j
                    anchored += 1
                break
    prev = -1e9
    for li in sorted(out):
        if out[li] < prev + 0.05:
            out[li] = prev + 0.05
        prev = out[li]
    return out, anchored


# ---------- API ----------
def align_words(vocals_path, text, synced_lyrics=None, ffmpeg="ffmpeg"):
    """Ritorna (words, lines, info) con words = [{'word','start','end','line'}]."""
    lines = clean_lines(text)
    words, line_of, first_of_line = [], [], {}
    for li, ln in enumerate(lines):
        for wi, w in enumerate(ln.split()):
            if wi == 0:
                first_of_line[li] = len(words)
            words.append(w)
            line_of.append(li)
    if not words:
        return [], [], {"ok": False, "reason": "testo vuoto"}

    work = tempfile.mkdtemp(prefix="lam_")
    try:
        import time as _t
        adir, ndir = os.path.join(work, "audio"), os.path.join(work, "annot")
        os.makedirs(adir); os.makedirs(ndir)
        wavp = os.path.join(adir, "S.wav")
        _LOG(f"[LAM] step 1/3 ffmpeg->wav ({len(words)} parole)...")
        _t0 = _t.time()
        subprocess.run([ffmpeg, "-y", "-i", vocals_path, "-ac", "1", "-ar", "22050", wavp],
                       capture_output=True)
        if not os.path.exists(wavp):
            return [], lines, {"ok": False, "reason": "conversione audio fallita"}
        try:
            _dur = os.path.getsize(wavp) / (22050 * 2)
        except Exception:
            _dur = 0.0
        _LOG(f"[LAM] step 1/3 ok in {_t.time()-_t0:.1f}s (~{_dur:.0f}s audio)")
        _LOG("[LAM] step 2/3 fonemizzazione (espeak)...")
        _t0 = _t.time()
        _write_annot(words, ndir)
        _LOG(f"[LAM] step 2/3 ok in {_t.time()-_t0:.1f}s")
        _LOG("[LAM] step 3/3 forward modello + allineamento (CPU: puo' essere lento)...")
        _t0 = _t.time()
        raw, score = _lam_word_times(adir, ndir)
        _LOG(f"[LAM] step 3/3 ok in {_t.time()-_t0:.1f}s (score={score:.1f})")
        if len(raw) != len(words):
            raw = (raw + [(raw[-1] if raw else (0, 0))] * len(words))[:len(words)]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    starts = [t[0] for t in raw]

    # LAM PURO: nessun VAD, nessuno snap onset. L'unica correzione sono i tempi
    # synced sulle PRIME PAROLE di riga (sotto).

    # === SYNCED: SOLO LA PRIMA PAROLA DI OGNI RIGA ===
    # Il tempo synced di LRCLIB indica quando ATTACCA la riga: si sposta SOLO la
    # prima parola, le altre restano dove le ha messe LAM.
    # DISCRIMINANTE: i synced si usano solo se combaciano almeno il 90% delle
    # prime parole; sotto quella soglia sono di un'altra versione -> si va avanti
    # senza synced.
    n_anch = 0
    sync_frac = 0.0
    if synced_lyrics:
        try:
            sl = parse_synced_lyrics(synced_lyrics)
            if sl:
                cur = {li: starts[first_of_line[li]] for li in range(len(lines))}
                newls, n_match = _anchor_lines_to_synced(lines, cur, sl)
                matched = [li for li in newls if abs(newls[li] - cur[li]) > 0.001]
                frac = len(matched) / max(1, len(lines))
                sync_frac = frac
                if frac < 0.90:
                    print(f"[LAM] synced non applicati: combaciano {len(matched)}/{len(lines)} "
                          f"prime parole ({frac*100:.0f}% < 90%)")
                else:
                    for li in matched:
                        wi = first_of_line[li]
                        t = newls[li]
                        # non deve scavalcare le parole vicine (solo questa si muove)
                        if wi + 1 < len(starts):
                            t = min(t, starts[wi + 1] - 0.05)
                        if wi > 0:
                            t = max(t, starts[wi - 1] + 0.05)
                        starts[wi] = max(0.0, t)
                    n_anch = len(matched)
                    print(f"[LAM] synced applicati alle prime parole di {n_anch}/{len(lines)} righe "
                          f"({frac*100:.0f}%)")
        except Exception:
            n_anch = 0

    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1] + 0.02:
            starts[i] = starts[i - 1] + 0.02

    out = []
    for i in range(len(words)):
        st = max(0.0, starts[i])
        en = starts[i + 1] if i + 1 < len(starts) else st + 0.35
        out.append({"word": words[i], "start": round(st, 3),
                    "end": round(max(st + 0.05, en), 3), "line": line_of[i]})
    gaps = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
    big = [g for g in gaps if g > 10.0]
    span = (starts[-1] - starts[0]) if len(starts) > 1 else 0.0
    ok = not (len(big) > 3 or (span > 0 and sum(big) > 0.35 * span))
    info = {"ok": ok, "score": round(score, 1), "anchored": n_anch,
            "sync_frac": round(sync_frac, 3),
            "big_gaps": [round(g, 1) for g in big],
            "reason": "ok" if ok else "il testo non copre tutto il canto (probabile testo incompleto)"}
    return out, lines, info


def _genius_token():
    """Token API Genius: da variabile d'ambiente GENIUS_TOKEN o da dipendenze/config.json.
    Gratuito, si crea in 2 minuti su https://genius.com/api-clients"""
    tok = os.environ.get("GENIUS_TOKEN", "").strip()
    if tok:
        return tok
    try:
        import json as _json
        cfg = os.path.join(_HERE, "dipendenze", "config.json")
        if os.path.exists(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                return str(_json.load(f).get("genius_token") or "").strip()
    except Exception:
        pass
    return ""


def fetch_genius_lyrics(artist, title, token=None):
    """Testo da GENIUS tramite API ufficiale (serve token: la pagina pubblica
    risponde 403 alle richieste automatiche). L'API trova il brano e il suo URL,
    il testo si legge dalla pagina del brano."""
    import requests
    token = token or _genius_token()
    if not token:
        print("[GENIUS] nessun token (GENIUS_TOKEN o config.json 'genius_token') -> salto")
        return None
    try:
        r = requests.get("https://api.genius.com/search",
                         params={"q": f"{artist} {title}"},
                         headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code != 200:
            print(f"[GENIUS] search HTTP {r.status_code}")
            return None
        hits = (r.json().get("response") or {}).get("hits") or []
        kt, ka = _norm(title), _norm(artist)
        url = None
        for h in hits:
            res = h.get("result") or {}
            t = _norm(res.get("title") or "")
            a = _norm((res.get("primary_artist") or {}).get("name") or "")
            if kt[:12] in t and (ka[:6] in a or a[:6] in ka):
                url = res.get("url")
                break
        if not url and hits:
            url = (hits[0].get("result") or {}).get("url")
        if not url:
            return None
        # I testi NON sono nell'API e la pagina risponde 403 alle richieste
        # automatiche. Si usa invece l'EMBED UFFICIALE (pensato per essere
        # incorporato in siti terzi): genius.com/songs/<id>/embed.js
        song_id = None
        for h in hits:
            res = h.get("result") or {}
            if res.get("url") == url:
                song_id = res.get("id")
                break
        if not song_id:
            return None
        import codecs, html as _html
        em = requests.get(f"https://genius.com/songs/{song_id}/embed.js", timeout=20)
        if em.status_code != 200:
            print(f"[GENIUS] embed HTTP {em.status_code}")
            return None
        docs = []
        for m in re.finditer(r"JSON\.parse\('", em.text):
            start = m.end()
            end = em.text.find("')", start)
            if end <= start:
                continue
            s = em.text[start:end].replace("\\'", "'")
            try:
                docs.append(codecs.decode(s, "unicode_escape"))
            except Exception:
                docs.append(s)
        if not docs:
            return None
        doc = max(docs, key=len)
        body = re.search(r'rg_embed_body["\']?\s*>(.*)', doc, re.S)
        seg = body.group(1) if body else doc
        seg = seg.replace("\\n", "\n").replace("\\/", "/")
        seg = re.sub(r"<br\s*/?>", "\n", seg)
        seg = re.sub(r"<[^>]+>", "", seg)
        seg = _html.unescape(seg)
        seg = re.sub(r"\[[^\]]*\]", "", seg)              # [Strofa 1], [Ritornello]
        seg = re.sub(r"(?im)^\s*powered by genius\s*$", "", seg)
        # scarta righe di sola punteggiatura/virgolette (residui dell'escape)
        txt = "\n".join(l.strip() for l in seg.splitlines()
                        if l.strip() and any(ch.isalnum() for ch in l))
        if txt:
            print(f"[GENIUS] testo trovato (embed ufficiale): {len(txt.split())} parole")
        return txt or None
    except Exception as e:
        print(f"[GENIUS] errore: {e}")
        return None


def genius_canonical(artist, title):
    """Artista/titolo CANONICI da Genius (API ufficiale, col token).
    I testi di Genius NON sono prendibili (genius.com risponde 403 alle richieste
    automatiche e l'API non espone i lyrics), ma i metadati sì: servono a correggere
    nomi storpiati e a ricercare meglio su LRCLIB / lyrics.ovh."""
    import requests
    token = _genius_token()
    if not token:
        return None
    try:
        r = requests.get("https://api.genius.com/search",
                         params={"q": f"{artist} {title}"},
                         headers={"Authorization": f"Bearer {token}"}, timeout=20)
        if r.status_code != 200:
            print(f"[GENIUS] search HTTP {r.status_code}")
            return None
        hits = (r.json().get("response") or {}).get("hits") or []
        kt = _norm(title)
        for h in hits:
            res = h.get("result") or {}
            t = res.get("title") or ""
            a = (res.get("primary_artist") or {}).get("name") or ""
            if not t or not a:
                continue
            if _norm(t)[:10] in kt or kt[:10] in _norm(t):
                if (_norm(t), _norm(a)) != (_norm(title), _norm(artist)):
                    print(f"[GENIUS] metadati canonici: '{a}' - '{t}'")
                return a, t
        return None
    except Exception as e:
        print(f"[GENIUS] errore: {e}")
        return None


def fetch_lyricsovh(artist, title):
    """Testo da lyrics.ovh (API pubblica gratuita, nessun token)."""
    import requests
    try:
        r = requests.get(f"https://api.lyrics.ovh/v1/{artist}/{title}", timeout=20)
        if r.status_code != 200:
            return None
        txt = (r.json().get("lyrics") or "").strip()
        txt = txt.replace("\r\n", "\n")
        txt = re.sub(r"^\s*Paroles de la chanson.*$", "", txt, flags=re.M)
        txt = "\n".join(l.strip() for l in txt.splitlines() if l.strip())
        if txt:
            print(f"[LYRICS.OVH] testo trovato: {len(txt.split())} parole")
        return txt or None
    except Exception as e:
        print(f"[LYRICS.OVH] errore: {e}")
        return None


# Archivio LRC LOCALE (fonte primaria sul PC, con i tempi synced): ~16.000 file
# in <lettera>/<artista>/<album>/<titolo>.lrc
LOCAL_LRC_DIR = os.path.join(_HERE, "addestramento", "datasets", "self_training", "songs", "lrc")
# DB blob dell'archivio (costruito da costruisci_lyrics_db.py): lookup istantanea
# invece di os.walk su ~100k file. Se manca, si torna alla scansione del disco.
LOCAL_LRC_DB = os.path.join(_HERE, "lyrics_local.db")
_lrc_db = None  # connessione sqlite in cache


def _get_lrc_db():
    global _lrc_db
    if _lrc_db is False:
        return None
    if _lrc_db is None:
        try:
            import sqlite3
            if not os.path.exists(LOCAL_LRC_DB):
                _lrc_db = False
                return None
            _lrc_db = sqlite3.connect(f"file:{LOCAL_LRC_DB}?mode=ro", uri=True,
                                      check_same_thread=False)
        except Exception as e:
            print(f"[LOCALE] DB non apribile ({e}) -> uso il disco")
            _lrc_db = False
            return None
    return _lrc_db


def _fetch_local_lrc_db(artist, title):
    """Come fetch_local_lrc ma via DB blob. Ritorna (plain, synced) o None se il DB
    non c'e'. Riproduce lo stesso match (cartella-artista + nome-file normalizzati)."""
    con = _get_lrc_db()
    if con is None:
        return None
    ka = _norm(artist)
    kt = _norm(title)
    kt_key = re.sub(r"\s*\(.*?\)\s*", " ", kt).strip()
    if not ka or not kt_key:
        return (None, None)
    try:
        # 1) match ESATTO (usa l'indice ix_akt -> istantaneo, copre la maggioranza)
        rows = con.execute(
            "SELECT lrc FROM lyrics WHERE artist_key=? AND title_key=? "
            "ORDER BY has_synced DESC LIMIT 20", (ka, kt_key)).fetchall()
        if not rows:
            # 2) fallback substring: artista ka in/di na ; titolo kt_key in n | n_key in kt
            rows = con.execute(
                "SELECT lrc FROM lyrics WHERE "
                "(artist_key=? OR instr(artist_key,?)>0 OR instr(?,artist_key)>0) AND "
                "(title_key=? OR instr(title_norm,?)>0 OR instr(?,title_key)>0) "
                "ORDER BY has_synced DESC LIMIT 20",
                (ka, ka, ka, kt_key, kt_key, kt)).fetchall()
    except Exception as e:
        print(f"[LOCALE] errore DB ({e}) -> uso il disco")
        return None
    for (raw,) in rows:
        txt = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw))
        plain, synced = _lrc_split(txt)
        if plain:
            print(f"[LOCALE-DB] trovato: {artist} - {title} "
                  f"({len(plain.split())} parole{', + synced' if synced else ''})")
            return plain, (synced or None)
    return (None, None)


def _lrc_split(lrc_text):
    """Da un .lrc restituisce (plain, synced): plain = solo testo, synced = solo le
    righe con timestamp [mm:ss.xx]."""
    plain_lines, synced_lines = [], []
    for ln in str(lrc_text or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("[ar:") or s.startswith("[ti:") or s.startswith("[al:") \
           or s.startswith("[id:") or s.startswith("[by:") or s.startswith("[length:"):
            continue
        m = re.match(r"\s*\[(\d+):(\d+(?:[.,]\d+)?)\]\s*(.*)", s)
        if m:
            txt = (m.group(3) or "").strip()
            if txt:
                synced_lines.append(s)
                plain_lines.append(txt)
        else:
            plain_lines.append(re.sub(r"^\s*\[[^\]]*\]\s*", "", s).strip())
    return "\n".join(p for p in plain_lines if p), "\n".join(synced_lines)


def _fix_mojibake(text):
    """Ripara i testi UTF-8 letti come Latin-1 (bÃ©bÃ© -> bébé)."""
    if not text:
        return text
    if any(m in text for m in ("Ã", "Â", "Å", "â€")):
        try:
            fixed = text.encode("latin-1").decode("utf-8")
            if fixed.count("Ã") < text.count("Ã"):
                return fixed
        except Exception:
            pass
    return text


def _artist_match(query, cand):
    """L'artista del candidato è compatibile con quello richiesto? (senza artista in
    query -> non posso filtrare, accetto). Evita di prendere 'YL - Sicario' per
    ''A67 - 'O sicario' solo perché il titolo combacia."""
    if not query:
        return True
    a, b = _norm(query), _norm(cand or "")
    if not b:
        return False
    if a == b or a in b or b in a:
        return True
    ta = {t for t in re.split(r"\W+", a) if len(t) >= 3}
    tb = {t for t in re.split(r"\W+", b) if len(t) >= 3}
    return bool(ta & tb)


# marcatori inequivocabili di francese/inglese (per scartare il brano sbagliato)
_WRONG_LANG = re.compile(
    r"\b(j'|qu'|n'|c'est|dans|avec|pour|toujours|jamais|les|des|une|mais|est|"
    r"the|you|your|and|don't|i'm|gonna|baby|yeah|ain't)\b", re.I)


def _looks_wrong_language(text):
    """True se il testo NON è italiano/napoletano -> è la canzone sbagliata (polacco,
    francese, inglese...). Usa langdetect (il napoletano viene rilevato come 'it');
    se manca, ripiega sui marcatori FR/EN. Conservativo: se incerto NON scarta."""
    if not text or len(text.split()) < 6:
        return False
    try:
        from langdetect import detect_langs
        langs = detect_langs(text)
        if langs:
            top = langs[0]
            return top.lang != "it" and top.prob >= 0.85
        return False
    except Exception:
        n = len(_WRONG_LANG.findall(text))
        return n >= 8 and (n / max(1, len(text.split()))) > 0.06


def fetch_lrclib(artist, title, duration=None):
    """Testo da LRCLIB (API pubblica gratuita, nessun token). Ha SPESSO i synced.
    Ritorna (plain, synced) oppure (None, None). Match esatto, poi ricerca FILTRATA
    per artista compatibile (non solo titolo!) e con durata vicina."""
    import requests
    hdr = {"User-Agent": "KaraDom (https://karadom.it)"}
    try:
        # 1) match esatto (LRCLIB /get filtra gia' artista+titolo)
        if artist and title:
            r = requests.get("https://lrclib.net/api/get",
                             params={"artist_name": artist, "track_name": title},
                             headers=hdr, timeout=20)
            if r.status_code == 200:
                j = r.json() or {}
                syn = _fix_mojibake((j.get("syncedLyrics") or "").strip())
                pla = _fix_mojibake((j.get("plainLyrics") or "").strip())
                if pla or syn:
                    print(f"[LRCLIB] match esatto ({len((pla or syn).split())} parole"
                          f"{', + synced' if syn else ''})")
                    return (pla or None), (syn or None)
        # 2) ricerca -> FILTRA per artista compatibile (evita gli omonimi di altri artisti)
        params = {"track_name": title}
        if artist:
            params["artist_name"] = artist
        r = requests.get("https://lrclib.net/api/search", params=params,
                         headers=hdr, timeout=20)
        if r.status_code != 200:
            return None, None
        cands = r.json() or []
        if artist:
            filtrati = [c for c in cands if _artist_match(artist, c.get("artistName"))]
            if not filtrati:
                print(f"[LRCLIB] scartato: nessun match d'ARTISTA per '{artist}' "
                      f"(trovati altri artisti col titolo '{title}')")
                return None, None
            cands = filtrati
        if not cands:
            return None, None

        def _score(c):
            s = 0
            if c.get("syncedLyrics"):
                s += 1000
            if duration and c.get("duration"):
                s -= abs(float(c["duration"]) - float(duration))
            return s
        best = max(cands, key=_score)
        syn = _fix_mojibake((best.get("syncedLyrics") or "").strip())
        pla = _fix_mojibake((best.get("plainLyrics") or "").strip())
        if pla or syn:
            print(f"[LRCLIB] ricerca: '{best.get('artistName')}' - '{best.get('trackName')}' "
                  f"({len((pla or syn).split())} parole{', + synced' if syn else ''})")
            return (pla or None), (syn or None)
    except Exception as e:
        print(f"[LRCLIB] errore: {e}")
    return None, None


def _elevenlabs_key():
    k = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if k:
        return k
    try:
        import json as _json
        cfg = os.path.join(_HERE, "dipendenze", "config.json")
        if os.path.exists(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                return str(_json.load(f).get("elevenlabs_api_key") or "").strip()
    except Exception:
        pass
    return ""


def fetch_scribe(audio_path, language="ita"):
    """ULTIMA SPIAGGIA: trascrive il CANTATO con ElevenLabs Scribe (scribe_v1) quando
    non c'e' testo da nessuna fonte e l'utente non lo incolla. Il testo trascritto
    combacia col canto reale (niente discrepanze) -> il LAM lo allinea bene.
    A PAGAMENTO (~0,015 $/brano da 4 min): serve la key ELEVENLABS_API_KEY o
    config.json 'elevenlabs_api_key'. Ritorna il testo (righe spezzate sui silenzi) o None."""
    import requests
    key = _elevenlabs_key()
    if not key:
        print("[SCRIBE] nessuna API key (ELEVENLABS_API_KEY o config.json "
              "'elevenlabs_api_key') -> salto")
        return None
    try:
        with open(audio_path, "rb") as fh:
            r = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": key},
                data={"model_id": "scribe_v1", "language_code": language},
                files={"file": (os.path.basename(audio_path), fh, "audio/wav")},
                timeout=300)
        if r.status_code != 200:
            print(f"[SCRIBE] HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json() or {}
        words = data.get("words") or []
        # ricostruisci le righe: nuova riga quando c'e' un silenzio > 0.7s tra parole
        lines, cur, prev_end = [], [], None
        for w in words:
            if w.get("type") not in (None, "word"):
                continue
            wt = (w.get("text") or "").strip()
            if not wt:
                continue
            st = w.get("start")
            if prev_end is not None and st is not None and (st - prev_end) > 0.7 and cur:
                lines.append(" ".join(cur)); cur = []
            cur.append(wt)
            prev_end = w.get("end", st)
        if cur:
            lines.append(" ".join(cur))
        txt = "\n".join(lines).strip() or (data.get("text") or "").strip()
        if txt:
            print(f"[SCRIBE] trascritto: {len(txt.split())} parole, {len(lines)} righe")
        return txt or None
    except Exception as e:
        print(f"[SCRIBE] errore: {e}")
        return None


def save_local_lrc(artist, title, plain, synced=None, source="dump"):
    """Salva un testo nell'archivio LRC locale (per costruire il dump da Genius/
    lyrics.ovh). Se ci sono i synced li scrive come [mm:ss.xx], altrimenti solo il
    testo. Non sovrascrive un file gia' esistente."""
    if not plain or not artist or not title:
        return None
    def _safe(s):
        return re.sub(r'[<>:"/\\|?*]', "_", str(s)).strip()[:120] or "x"
    ka = _norm(artist)
    grp = "0-H" if (ka[:1] < "i") else ("I-Q" if ka[:1] < "r" else "R-Z")
    d = os.path.join(LOCAL_LRC_DIR, grp, _safe(artist), "_dump")
    path = os.path.join(d, _safe(title) + ".lrc")
    if os.path.exists(path):
        return path
    try:
        os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"[ar:{artist}]\n[ti:{title}]\n[by:{source}]\n")
            if synced:
                f.write(synced.strip() + "\n")
            else:
                for ln in plain.splitlines():
                    if ln.strip():
                        f.write(ln.strip() + "\n")
        print(f"[LOCALE] salvato in archivio: {os.path.basename(path)} (fonte {source})")
        try:
            with open(path, "rb") as f:
                _db_add(artist, title, f.read())
        except Exception:
            pass
        return path
    except Exception as e:
        print(f"[LOCALE] salvataggio fallito: {e}")
        return None


def _db_add(artist, title, raw_bytes):
    """Inserisce un nuovo brano nel DB blob (se esiste), cosi' fetch_local_lrc lo
    trova subito senza dover rigenerare tutto il DB."""
    if not os.path.exists(LOCAL_LRC_DB):
        return
    try:
        import sqlite3
        txt = raw_bytes.decode("utf-8", "replace") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
        _plain, _syn = _lrc_split(txt)
        ak = _norm(artist)
        tn = _norm(title)
        tk = re.sub(r"\s*\(.*?\)\s*", " ", tn).strip()
        con = sqlite3.connect(LOCAL_LRC_DB, timeout=15)
        con.execute(
            "INSERT INTO lyrics(artist,title,artist_key,title_norm,title_key,"
            "has_synced,path,lrc) VALUES(?,?,?,?,?,?,?,?)",
            (artist, title, ak, tn, tk, 1 if _syn else 0, "_dump", raw_bytes))
        con.commit()
        con.close()
        print(f"[DB] +1 brano nel DB blob: {artist} - {title}")
    except Exception as e:
        print(f"[DB] insert fallito: {e}")


def guess_artist_by_title(title):
    """Quando il file non ha l'artista (es. scaricato da YouTube: 'Titolo [id]'),
    prova a INDOVINARLO dal DB blob: cerca chi, in archivio, ha un brano con questo
    titolo. Ritorna il nome artista o None. Preferisce match con synced e piu' diffusi."""
    con = _get_lrc_db()
    if con is None:
        return None
    kt = _norm(title)
    kt_key = re.sub(r"\s*\(.*?\)\s*", " ", kt).strip()
    if not kt_key or len(kt_key) < 3:
        return None
    try:
        # 1) titolo IDENTICO in archivio -> artista piu' probabile (synced, poi frequenza)
        row = con.execute(
            "SELECT artist FROM lyrics WHERE title_key=? "
            "GROUP BY artist_key ORDER BY MAX(has_synced) DESC, COUNT(*) DESC LIMIT 1",
            (kt_key,)).fetchone()
        if not row:
            # 2) fallback: un titolo d'archivio (>=4 char) contenuto nel nostro nome
            #    (il nome file ha rumore extra) -> prendi il match piu' lungo/specifico
            row = con.execute(
                "SELECT artist FROM lyrics WHERE instr(?, title_key)>0 "
                "AND length(title_key)>=4 "
                "ORDER BY has_synced DESC, length(title_key) DESC LIMIT 1",
                (kt_key,)).fetchone()
    except Exception:
        return None
    return row[0] if row else None


def fetch_local_lrc(artist, title):
    """Cerca il brano nell'archivio LRC LOCALE. Ritorna (plain, synced) oppure
    (None, None). Match: cartella artista + file/riga titolo (normalizzati)."""
    # 0) DB blob (istantaneo). Se il DB c'e', decide lui; solo se manca si va sul disco.
    db = _fetch_local_lrc_db(artist, title)
    if db is not None:
        plain, synced = db
        if plain:
            plain = _fix_mojibake(plain)
            synced = _fix_mojibake(synced) if synced else synced
            # anche un testo GIA' in archivio puo' essere della canzone sbagliata
            # (salvato per errore in passato): scartalo -> si ripiega su LRCLIB/Genius.
            if _looks_wrong_language(plain):
                print("[LOCALE-DB] scartato: testo in un'altra lingua (canzone sbagliata)")
                return None, None
            return plain, synced
        return db

    if not os.path.isdir(LOCAL_LRC_DIR):
        return None, None
    ka, kt = _norm(artist), _norm(title)
    kt_key = re.sub(r"\s*\(.*?\)\s*", " ", kt).strip()  # togli "(feat...)" ecc.

    def match_title(fname):
        n = _norm(os.path.splitext(fname)[0])
        n_key = re.sub(r"\s*\(.*?\)\s*", " ", n).strip()
        return kt_key and (kt_key == n_key or kt_key in n or n_key in kt)

    # 1) cerca la cartella artista nelle sottocartelle-lettera, poi il file titolo
    try:
        for grp in os.scandir(LOCAL_LRC_DIR):
            if not grp.is_dir():
                continue
            for art in os.scandir(grp.path):
                if not art.is_dir():
                    continue
                na = _norm(art.name)
                if not (ka and (ka == na or ka in na or na in ka)):
                    continue
                for root, _dirs, files in os.walk(art.path):
                    for f in files:
                        if f.lower().endswith(".lrc") and match_title(f):
                            try:
                                txt = open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
                            except Exception:
                                continue
                            plain, synced = _lrc_split(txt)
                            if plain:
                                print(f"[LOCALE] trovato: {art.name} - {f} "
                                      f"({len(plain.split())} parole{', + synced' if synced else ''})")
                                return plain, (synced or None)
    except Exception as e:
        print(f"[LOCALE] errore: {e}")
    return None, None


def fetch_lyrics_candidates(artist, title, audio_duration=None, limit=5):
    """SUL PC: fonte PRIMARIA = archivio LRC LOCALE (con synced); FALLBACK = Genius
    e lyrics.ovh. NIENTE LRCLIB. Ritorna (lista_testi, synced)."""
    out = []
    if not artist or not title:
        return out, None

    # 1) ARCHIVIO LOCALE (prima scelta: offline, e ha i tempi synced)
    best_synced = None
    local_plain, local_synced = fetch_local_lrc(artist, title)
    if local_plain:
        out.append(local_plain)
        best_synced = local_synced

    seen_full = {" ".join(p.split()).lower() for p in out}

    # 1b) LRCLIB ("ovunque"): ha spesso i synced. Se l'archivio non aveva i tempi,
    #     li prende da qui.
    try:
        lp, ls = fetch_lrclib(artist, title, audio_duration)
        if lp and _looks_wrong_language(lp):
            print("[LRCLIB] scartato: il testo sembra un'altra lingua (canzone sbagliata)")
            lp, ls = None, None
        if lp:
            k = " ".join(lp.split()).lower()
            if k not in seen_full:
                out.append(lp); seen_full.add(k)
        if ls and not best_synced:
            best_synced = ls
    except Exception as e:
        print(f"[LRCLIB] errore: {e}")

    # GENIUS: i testi non sono accessibili (403 Cloudflare), ma l'API da' i nomi
    # CANONICI -> se differiscono, si ritenta lyrics.ovh anche con quelli.
    _pairs = [(artist, title)]
    try:
        can = genius_canonical(artist, title)
        if can and (_norm(can[0]), _norm(can[1])) != (_norm(artist), _norm(title)):
            _pairs.append(can)
    except Exception:
        pass

    sources = [(fetch_genius_lyrics, "GENIUS", (artist, title))]
    sources += [(fetch_lyricsovh, "LYRICS.OVH", p) for p in _pairs]
    for _fn, _label, (_a, _t) in sources:
        try:
            extra = _fn(_a, _t)
        except Exception as _e:
            print(f"[{_label}] errore: {_e}")
            extra = None
        if not extra:
            continue
        extra = _fix_mojibake(extra)
        if _looks_wrong_language(extra):
            print(f"[{_label}] scartato: il testo sembra un'altra lingua (canzone sbagliata)")
            continue
        k = " ".join(extra.split()).lower()
        if k in seen_full:
            print(f"[{_label}] testo identico a un candidato gia' presente -> ignorato")
            continue
        seen_full.add(k)
        out.append(extra)
        # CACHE/DUMP: se il testo non veniva dall'archivio locale, salvalo li' cosi'
        # l'archivio cresce da solo (la prossima volta niente rete).
        if not local_plain:
            save_local_lrc(artist, title, extra, synced=best_synced, source=_label)

    print(f"[TESTI] {len(out)} candidati totali (LOCALE + Genius + lyrics.ovh): "
          f"{[len(p.split()) for p in out]} parole")
    return out, best_synced


def align_best(vocals_path, artist=None, title=None, audio_duration=None,
               text=None, ffmpeg="ffmpeg"):
    """Allineamento 'come il sito': se `text` e' fornito (incollato dall'utente) ha
    la precedenza; altrimenti cerca PIU' testi su LRCLIB e prova finche' uno COPRE
    davvero il canto (check di copertura). Ritorna (lrc, words, info, testo_usato)."""
    synced = None
    if text and str(text).strip():
        cands = [text]
        try:
            _, synced = fetch_lyrics_candidates(artist, title, audio_duration)
        except Exception:
            synced = None
    else:
        cands, synced = fetch_lyrics_candidates(artist, title, audio_duration)
    # NB (come il SITO): si usa il TESTO COMPLETO (piu' ricco, che copre il canto) +
    # ancore synced dove ci sono + RIESCALATURA tra le ancore per le righe senza
    # synced. NON si sostituisce il testo con quello (piu' corto) del synced.

    if not cands:
        return "", [], {"ok": False, "reason": "nessun testo trovato per il brano"}, ""

    last = {"ok": False, "reason": "nessun testo copre il canto"}
    for i, cand in enumerate(cands, 1):
        lrc, words, info = align_to_lrc(vocals_path, cand, synced_lyrics=synced, ffmpeg=ffmpeg)
        # Il testo va bene se COPRE il canto e, quando ci sono i synced, se le sue
        # prime parole ci combaciano per almeno il 90% (altrimenti e' un'altra
        # versione: si prova il testo successivo).
        # PRIORITA': numero di parole e durata (cioe' il check di copertura)
        # decidono QUALE testo usare. I synced vengono DOPO: si applicano al testo
        # scelto solo se le prime parole combaciano dal 90% in su.
        if info.get("ok") and words:
            print(f"[LAM] testo {i}/{len(cands)} OK (synced sulle prime parole di "
                  f"{info.get('anchored', 0)} righe, {info.get('sync_frac',0)*100:.0f}%)")
            return lrc, words, info, cand
        print(f"[LAM] testo {i}/{len(cands)} SCARTATO: {info.get('reason')}")
        last = info
        if not lrc:
            continue
        last_good = (lrc, words, info, cand)
    # nessuno passa il check: uso comunque il primo che ha prodotto tempi
    try:
        return last_good[0], last_good[1], last_good[2], last_good[3]
    except Exception:
        return "", [], last, ""


def _fmt(sec):
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def align_to_lrc(vocals_path, text, synced_lyrics=None, ffmpeg="ffmpeg"):
    """Allinea e produce un LRC ENHANCED word-level identico nel formato a quello
    che il programma si aspetta:  [mm:ss.xx]<mm:ss.xx>PAROLA <mm:ss.xx>PAROLA
    Ritorna (lrc_text, words, info)."""
    words, lines, info = align_words(vocals_path, text, synced_lyrics, ffmpeg)
    if not words:
        return "", [], info
    by_line = {}
    for w in words:
        by_line.setdefault(w["line"], []).append(w)
    out = []
    for li in sorted(by_line):
        ws = by_line[li]
        head = _fmt(ws[0]["start"])
        body = " ".join(f"<{_fmt(w['start'])}>{w['word']}" for w in ws)
        out.append(f"[{head}]{body}")
    return "\n".join(out), words, info
