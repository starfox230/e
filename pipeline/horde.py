"""A small client for AI Horde (stablehorde.net), the volunteer-run GPU network.

This container has no GPU, and a one-step model on four CPU cores produces waxy faces and
drops half of what a scene asks for. The Horde runs full models -- FLUX.1-schnell,
Z-Image-Turbo -- on volunteers' GPUs, free, with anonymous access at the lowest queue
priority. Requests are submitted asynchronously and polled; nothing is shared to the Horde's
public dataset.
"""
import io
import json
import time
import urllib.request

API = 'https://stablehorde.net/api/v2'
ANON = '0000000000'
AGENT = 'umbrella-film:1.0:anonymous'


def _req(method, path, body=None, key=ANON, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={'apikey': key, 'Client-Agent': AGENT,
                                        'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)


MODELS = {
    'flux': {'models': ['Flux.1-Schnell fp8 (Compact)'], 'steps': 4, 'cfg_scale': 1.0, 'sampler_name': 'k_euler'},
    'zimage': {'models': ['Z-Image-Turbo'], 'steps': 8, 'cfg_scale': 1.0, 'sampler_name': 'k_euler'},
}


def submit(prompt, seed, model='flux', width=1024, height=576, key=ANON):
    m = MODELS[model]
    body = {
        'prompt': prompt,
        'params': {'width': width, 'height': height, 'steps': m['steps'], 'cfg_scale': m['cfg_scale'],
                   'sampler_name': m['sampler_name'], 'seed': str(seed), 'n': 1, 'karras': False},
        'models': m['models'],
        'r2': True,
        'shared': False,          # keep the story's frames out of the Horde's public dataset
        'nsfw': False,
        'censor_nsfw': True,
        'slow_workers': True,
        'trusted_workers': False,
    }
    return _req('POST', '/generate/async', body, key)['id']


def check(rid):
    return _req('GET', f'/generate/check/{rid}')


def fetch(rid):
    """The finished image as a PIL image, or None if the request faulted or was censored."""
    from PIL import Image
    st = _req('GET', f'/generate/status/{rid}')
    gens = st.get('generations') or []
    if not gens or gens[0].get('censored'):
        return None, st
    url = gens[0]['img']
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': AGENT}), timeout=120) as r:
        return Image.open(io.BytesIO(r.read())).convert('RGB'), gens[0]


def generate(prompt, seed, model='flux', width=1024, height=576, key=ANON, poll=4, patience=1800):
    rid = submit(prompt, seed, model, width, height, key)
    t0 = time.time()
    while time.time() - t0 < patience:
        c = check(rid)
        if c.get('faulted'):
            return None, {'faulted': True}
        if c.get('done'):
            return fetch(rid)
        time.sleep(poll)
    return None, {'timeout': True}
