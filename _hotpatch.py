# -*- coding: utf-8 -*-
# HOTPATCH loader — semina UNA volta (via build); da lì in poi le modifiche di LOGICA
# al worker NON richiedono un rebuild dell'immagine. All'avvio del container scarica
# i file dell'allowlist dal VPS (cartella pubblica servita da Apache) e sovrascrive
# quelli baked in /app/. Fonte = VPS (NON GitHub: un push su GitHub farebbe ripartire
# il build, vanificando tutto). Fail-safe: se il download manca/fallisce o il codice
# e' rotto (ast.parse), tiene la versione gia' nell'immagine. Nessun segreto.
#
# Per applicare una patch (senza build): carica il nuovo file in
#   /opt/karaokecreator/public/worker_hotpatch/<nome>.py  sul VPS.
# Al prossimo cold-start del worker viene preso da solo.
#
# Toggle:  WORKER_HOTPATCH=0            -> disattiva (usa solo il baked)
# URL:     WORKER_HOTPATCH_URL=<base>   -> cambia la sorgente (default sotto)
import os as _hp_os
import ast as _hp_ast
import hashlib as _hp_hash
import urllib.request as _hp_req

_HP_BASE = _hp_os.environ.get(
    "WORKER_HOTPATCH_URL",
    "https://karaokecreator.karadom.it/worker_hotpatch",
).rstrip("/")
# Solo LOGICA. NON mettere qui rp_handler.py (contiene il redirect FTP montato in build).
_HP_FILES = ["pipeline.py"]


def _hp_apply():
    for _f in _HP_FILES:
        _url = _HP_BASE + "/" + _f
        try:
            _rq = _hp_req.Request(_url, headers={"User-Agent": "KaraokeCreator-hotpatch/1.0"})
            _data = _hp_req.urlopen(_rq, timeout=15).read()
        except Exception as _e:
            print("[HOTPATCH] %s: nessuna patch (%s) -> tengo il baked" % (_f, _e), flush=True)
            continue
        if not _data or len(_data) < 50:
            print("[HOTPATCH] %s: risposta vuota/corta -> tengo il baked" % _f, flush=True)
            continue
        try:
            _txt = _data.decode("utf-8")
            _hp_ast.parse(_txt)  # non applicare MAI codice che non compila
        except Exception as _e:
            print("[HOTPATCH] %s: patch NON valida (%s) -> tengo il baked" % (_f, _e), flush=True)
            continue
        _dst = "/app/" + _f
        try:
            _old = open(_dst, encoding="utf-8").read()
        except Exception:
            _old = ""
        if _old == _txt:
            print("[HOTPATCH] %s: uguale al baked, nessun cambio" % _f, flush=True)
            continue
        try:
            with open(_dst, "w", encoding="utf-8") as _w:
                _w.write(_txt)
            print("[HOTPATCH] %s AGGIORNATO dal VPS (%d byte, sha1=%s)"
                  % (_f, len(_data), _hp_hash.sha1(_data).hexdigest()[:10]), flush=True)
        except Exception as _e:
            print("[HOTPATCH] %s: scrittura fallita (%s) -> tengo il baked" % (_f, _e), flush=True)


try:
    if _hp_os.environ.get("WORKER_HOTPATCH", "1") == "1":
        _hp_apply()
    else:
        print("[HOTPATCH] disattivato (WORKER_HOTPATCH=0)", flush=True)
except Exception as _hp_e:
    print("[HOTPATCH] loader in errore, ignoro e proseguo: %s" % _hp_e, flush=True)
