"""KaraokeCreator, RISERVA RunPod della pagina nuova: STESSO metodo di Modal (D:/Claude/KCModal/modal_kc.py).
Si usa solo quando Modal non risponde (es. workspace disabilitato a credito finito).

  separazione con cori (hq) = strada J: strumenti = mix - voce BS-RoFormer anvuew; cori = voce Demucs htdemucs_ft
      - solista (karaoke aufr33 + gabox sulla voce Demucs); LEAD = quel solista.
      Su una scheda sola il brano NON si spezza in 3 pezzi (servivano solo a dividerlo fra 6 schede Modal):
      anvuew e Demucs girano insieme, poi i due karaoke insieme.
  con cori (std) -> lead = karaoke aufr33 sul mix, base = mix - lead
  senza cori -> lead = karaoke aufr33 + gabox, base = mix - voce anvuew
  allinea / trascrivi = w2v-bert-2.0-music-lyrics-ctc, stesso codice dell'Allineatore Modal.

Input: {"op": "separa"|"allinea"|"trascrivi"|"sveglia", "audio_url" | "audio_b64", "cori", "qualita", "parole",
        "ftp_host", "ftp_user", "ftp_pass", "ftp_dir", "public_base_url"}
Output separa: {"download_url": zip (lead.mp3, base.mp3, tempi.json) caricato via FTP sul VPS, "tempi": {...}}
Output allinea / trascrivi: lo stesso dizionario dell'Allineatore Modal.
"""
import base64, ftplib, glob, io, json, os, subprocess, tempfile, threading, time, uuid, zipfile

import numpy as np
import requests
import runpod
import soundfile as sf

OVERLAP = '8'
MOD = '/opt/modelli'
CTC_SR = 16000
_UA = {'User-Agent': 'Mozilla/5.0 (compatible; KaraokeCreator/1.0)'}


# ---------------------------------------------------------------- separazione
def roformer(wav, m):
    t = time.time()
    d = tempfile.mkdtemp()
    ing, out = os.path.join(d, 'mix.wav'), os.path.join(d, 'out.wav')
    open(ing, 'wb').write(wav)
    r = subprocess.run(['bs_roformer-cli', '%s/%s.gguf' % (MOD, m), ing, out, '--overlap', OVERLAP],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError('bs_roformer %s: %s' % (m, (r.stdout + r.stderr)[-600:]))
    return open(out, 'rb').read(), round(time.time() - t, 2)


_DEMUCS = None
_DLOCK = threading.Lock()


def demucs_voce(wav):
    """Voce INTERA con Demucs v4 htdemucs_ft (ensemble di 4, shifts 2, overlap 0,5), audio-separator."""
    global _DEMUCS
    from audio_separator.separator import Separator
    t = time.time()
    d = tempfile.mkdtemp()
    ing = os.path.join(d, 'mix.wav')
    open(ing, 'wb').write(wav)
    with _DLOCK:
        if _DEMUCS is None:
            _DEMUCS = Separator(output_dir=d, model_file_dir=MOD + '/as', output_format='WAV',
                                demucs_params={'segment_size': 'Default', 'shifts': 2, 'overlap': 0.5,
                                               'segments_enabled': True})
            _DEMUCS.load_model(model_filename='htdemucs_ft.yaml')
        _DEMUCS.output_dir = d
        _DEMUCS.model_instance.output_dir = d
        _DEMUCS.separate(ing)
    return open(glob.glob(os.path.join(d, '*(Vocals)*'))[0], 'rb').read(), round(time.time() - t, 2)


def insieme(*lavori):
    """Lancia i lavori in parallelo sulla stessa scheda e ne aspetta i risultati (in ordine)."""
    ris, err = [None] * len(lavori), []

    def uno(i, f, a):
        try:
            ris[i] = f(*a)
        except Exception as e:
            err.append(e)
    th = [threading.Thread(target=uno, args=(i, f, a)) for i, (f, a) in enumerate(lavori)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if err:
        raise err[0]
    return ris


def separa(audio, cori, qualita):
    tempi = {'motore': 'runpod'}
    t0 = time.time()
    d = tempfile.mkdtemp()
    ing, wav = os.path.join(d, 'ingresso'), os.path.join(d, 'mix.wav')
    open(ing, 'wb').write(audio)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', ing, '-ac', '2', '-ar', '44100',
                    '-c:a', 'pcm_f32le', wav], check=True)
    mix, sr = sf.read(wav, dtype='float32', always_2d=True)
    tempi['decodifica_s'] = round(time.time() - t0, 2)
    tempi['durata_brano_s'] = round(len(mix) / sr, 2)
    lead_m = ['karaoke_aufr33', 'karaoke_gabox'] if qualita == 'hq' else ['karaoke_aufr33']
    voce_m = ['voce_anvuew'] if cori == 'senza' else []
    dati = open(wav, 'rb').read()
    rd = lambda b: sf.read(io.BytesIO(b), dtype='float32', always_2d=True)[0]

    def wavb(x):
        b = io.BytesIO()
        sf.write(b, x.astype(np.float32), sr, format='WAV', subtype='FLOAT')
        return b.getvalue()

    t1 = time.time()
    if cori == 'con' and qualita == 'hq':
        (ba, sa), (bd, sd) = insieme((roformer, (dati, 'voce_anvuew')), (demucs_voce, (dati,)))
        va, vd = rd(ba), rd(bd)
        tempi.update({'voce_anvuew_s': sa, 'demucs_s': sd, 'voci_s': round(time.time() - t1, 2)})
        t2 = time.time()
        n = min(len(mix), len(va), len(vd))
        vdb = wavb(vd[:n])
        usc = insieme(*[(roformer, (vdb, m)) for m in lead_m])
        n = min([n] + [len(rd(b)) for b, _ in usc])
        lead = np.mean([rd(b)[:n] for b, _ in usc], axis=0)
        tempi['karaoke_s'] = round(time.time() - t2, 2)
        tempi['separazione_s'] = round(time.time() - t1, 2)
        t2 = time.time()
        base = (mix[:n] - va[:n]) + (vd[:n] - lead)
        return consegna(d, sr, lead, base, tempi, t0, t2, cori, qualita, lead_m, ['voce_anvuew', 'htdemucs_ft'])

    usc = insieme(*[(roformer, (dati, m)) for m in lead_m + voce_m])
    stem = {m: rd(b) for m, (b, _) in zip(lead_m + voce_m, usc)}
    for m, (_, sec) in zip(lead_m + voce_m, usc):
        tempi['modello_%s_s' % m] = sec
    tempi['separazione_s'] = round(time.time() - t1, 2)
    t2 = time.time()
    n = min([len(mix)] + [len(v) for v in stem.values()])
    mix = mix[:n]
    lead = np.mean([stem[m][:n] for m in lead_m], axis=0)
    base = mix - (np.mean([stem[m][:n] for m in voce_m], axis=0) if cori == 'senza' else lead)
    return consegna(d, sr, lead, base, tempi, t0, t2, cori, qualita, lead_m, voce_m)


def consegna(d, sr, lead, base, tempi, t0, t2, cori, qualita, lead_m, voce_m):
    """lead.mp3 + base.mp3 (320k) + tempi.json in uno zip (come Modal)."""
    def mp3(x, nome):
        w = os.path.join(d, nome + '.wav')
        sf.write(w, np.clip(x, -1.0, 1.0), sr, subtype='PCM_24')
        o = os.path.join(d, nome + '.mp3')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', w, '-c:a', 'libmp3lame', '-b:a', '320k', o], check=True)
        return o
    files = dict(zip(('lead.mp3', 'base.mp3'), insieme((mp3, (lead, 'lead')), (mp3, (base, 'base')))))
    tempi['unione_e_mp3_s'] = round(time.time() - t2, 2)
    tempi['totale_s'] = round(time.time() - t0, 2)
    tempi.update({'cori': cori, 'qualita': qualita, 'overlap': OVERLAP,
                  'strada': 'J' if cori == 'con' and qualita == 'hq' else '',
                  'modelli_lead': lead_m, 'modelli_voce': voce_m})
    z = os.path.join(d, 'risultato.zip')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED) as f:
        for k, v in files.items():
            f.write(v, k)
        f.writestr('tempi.json', json.dumps(tempi, indent=1))
    print('[kc] %s' % json.dumps(tempi), flush=True)
    return z, tempi


def ftp_carica(inp, locale, nome):
    host, cartella = inp.get('ftp_host'), inp.get('ftp_dir') or '/risultati'
    if not (host and inp.get('ftp_user') and inp.get('public_base_url')):
        raise RuntimeError('destinazione FTP mancante nell input')
    ftp = ftplib.FTP(host, timeout=120)
    ftp.login(inp['ftp_user'], inp.get('ftp_pass') or '')
    try:
        ftp.cwd(cartella)
    except ftplib.error_perm:
        cur = ''
        for p in [p for p in cartella.split('/') if p]:
            cur += '/' + p
            try:
                ftp.cwd(cur)
            except ftplib.error_perm:
                ftp.mkd(cur)
                ftp.cwd(cur)
    with open(locale, 'rb') as f:
        ftp.storbinary('STOR ' + nome, f)
    ftp.quit()
    return inp['public_base_url'].rstrip('/') + '/' + nome


# ---------------------------------------------------------------- allineamento (CTC)
_CTC = {}


def ctc():
    if not _CTC:
        import torch
        from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
        dd = MOD + '/wbert'
        _CTC['proc'] = Wav2Vec2BertProcessor.from_pretrained(dd)
        _CTC['mod'] = Wav2Vec2BertForCTC.from_pretrained(dd).eval().cuda()
        _CTC['voc'] = _CTC['proc'].tokenizer.get_vocab()
    return _CTC


def _carica16(audio):
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, 'in'), os.path.join(d, 'in.wav')
    open(a, 'wb').write(audio)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a, '-ac', '1', '-ar', str(CTC_SR), b], check=True)
    return sf.read(b, dtype='float32')[0]


def _emissioni(x, core=30.0, ctx=2.0):
    """log-prob a 50 Hz su finestre da 30 s con 2 s di contorno per lato; si tiene il centro."""
    import torch
    m = ctc()
    c, k, out = int(core * CTC_SR), int(ctx * CTC_SR), []
    for i in range(0, len(x), c):
        a = max(0, i - k)
        inp = m['proc'](x[a:i + c + k], sampling_rate=CTC_SR, return_tensors='pt').to('cuda')
        with torch.no_grad():
            e = torch.log_softmax(m['mod'](**inp).logits[0].float(), -1).cpu().numpy()
        sx = (i - a) // 320
        n = int(np.ceil(min(c, len(x) - i) / 320))
        out.append(e[sx:sx + n])
    return np.concatenate(out, 0)


def allinea(audio, parole):
    import torch
    import torchaudio.functional as F
    t0 = time.time()
    x = _carica16(audio)
    lp = _emissioni(x)
    t_em = time.time() - t0
    voc = ctc()['voc']
    sep = voc['|']
    tok, quale = [sep], [-1]
    for i, w in enumerate(parole):
        for ch in [ch for ch in (w or '') if ch in voc and len(ch) == 1 and ch != '|']:
            tok.append(voc[ch]); quale.append(i)
        tok.append(sep); quale.append(-1)
    if len(tok) > lp.shape[0]:
        raise RuntimeError("testo troppo lungo per l'audio (%d lettere, %d fotogrammi)" % (len(tok), lp.shape[0]))
    al, sc = F.forced_align(torch.from_numpy(lp).unsqueeze(0), torch.tensor([tok], dtype=torch.int32), blank=0)
    spans = F.merge_tokens(al[0], sc[0].exp())
    res = [{'start': None, 'end': None, 'lettere': [], 'conf': None} for _ in parole]
    pr = [[] for _ in parole]
    for j, sp in enumerate(spans):
        w = quale[j]
        if w < 0:
            continue
        r = res[w]
        if r['start'] is None:
            r['start'] = round(sp.start * 0.02, 3)
        r['end'] = round((sp.end + 1) * 0.02, 3)
        r['lettere'].append(round(sp.start * 0.02, 3))
        pr[w].append(float(sp.score))
    for w, p in enumerate(pr):
        if p:
            res[w]['conf'] = round(sum(p) / len(p), 4)
    tutte = [q for p in pr for q in p]
    return {'parole': res, 'fiducia_media': round(sum(tutte) / max(1, len(tutte)), 4),
            'durata_s': round(len(x) / CTC_SR, 2), 'emissioni_s': round(t_em, 2),
            'totale_s': round(time.time() - t0, 2), 'modello': 'mrfakename/w2v-bert-2.0-music-lyrics-ctc'}


def trascrivi(audio):
    t0 = time.time()
    x = _carica16(audio)
    ids = np.argmax(_emissioni(x), -1).tolist()
    m = ctc()
    inv = {v: k for k, v in m['voc'].items()}
    blank = m['proc'].tokenizer.pad_token_id
    parole, cur, ini, fin, prec = [], '', None, None, None
    for f, i in enumerate(ids):
        if i != prec and i != blank:
            ch = inv.get(i, '')
            if ch in ('|', ' '):
                if cur:
                    parole.append({'word': cur, 'start': round(ini, 3), 'end': round(fin, 3)})
                    cur = ''
            elif len(ch) == 1:
                if not cur:
                    ini = f * 0.02
                cur += ch
        if i != blank and cur:
            fin = (f + 1) * 0.02
        prec = i
    if cur:
        parole.append({'word': cur, 'start': round(ini, 3), 'end': round(fin, 3)})
    return {'parole': parole, 'testo': ' '.join(p['word'] for p in parole),
            'durata_s': round(len(x) / CTC_SR, 2), 'totale_s': round(time.time() - t0, 2),
            'modello': 'mrfakename/w2v-bert-2.0-music-lyrics-ctc'}


# ---------------------------------------------------------------- RunPod
def _audio(inp):
    if inp.get('audio_url'):
        r = requests.get(inp['audio_url'], headers=_UA, timeout=300)
        r.raise_for_status()
        return r.content
    if inp.get('audio_b64'):
        return base64.b64decode(inp['audio_b64'])
    raise RuntimeError('audio mancante (audio_url / audio_b64)')


def handler(job):
    inp = job.get('input') or {}
    op = inp.get('op', 'separa')
    if op == 'sveglia':
        return {'ok': True}
    audio = _audio(inp)
    if op == 'allinea':
        return allinea(audio, [str(p) for p in inp.get('parole') or []])
    if op == 'trascrivi':
        return trascrivi(audio)
    cori = inp.get('cori') if inp.get('cori') in ('con', 'senza') else 'con'
    qualita = inp.get('qualita') if inp.get('qualita') in ('hq', 'std') else 'hq'
    z, tempi = separa(audio, cori, qualita)
    url = ftp_carica(inp, z, 'kdkc_%s_%s.zip' % (job.get('id', 'x')[:24], uuid.uuid4().hex[:8]))
    return {'download_url': url, 'tempi': tempi}


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
