"""SDXL-Turbo on OpenVINO, in NumPy: no torch, CPU only.

This container has no GPU and PyPI's torch drags in several gigabytes of CUDA libraries
it could never use, so the pipeline is written directly against the OpenVINO IR that
rupeshs/sdxl-turbo-openvino-int8 ships: two CLIP text encoders, an int8 UNet and the VAE
decoder. The sampler is the Euler-ancestral schedule SDXL-Turbo was distilled for, with
trailing timestep spacing -- one step goes straight from pure noise to the image.

SDXL-Turbo runs without classifier-free guidance, so there is no negative prompt: anything
the picture should not contain has to be kept out by how the prompt is worded.
"""
import os
import numpy as np
import openvino as ov
from PIL import Image

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work', 'models')
MODEL = os.path.join(MODELS, 'sdxl-turbo-ov')
TINY_VAE = os.path.join(MODELS, 'taesdxl-ov', 'vae_decoder', 'openvino_model.xml')
SR_MODEL = os.path.join(MODELS, 'sr', 'realesr-general-x4v3.onnx')
VAE_SCALE = 0.13025


def _sigmas():
    betas = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
    acp = np.cumprod(1.0 - betas)
    return ((1 - acp) / acp) ** 0.5


class SDXLTurbo:
    """768x448 by default. SDXL-Turbo was distilled at 512px; pushed to 1024x576 it starts
    tiling its subject -- a second head of hair beside the first, 'a lone scientist' drawn
    twice -- so it is run near its training size and the upscaler supplies the resolution.

    The decoder is the distilled tiny VAE. On identical latents it is indistinguishable from
    the full SDXL VAE at viewing size and 10-18x faster; the full VAE was 58% of the cost."""

    def __init__(self, width=768, height=448, steps=1, threads=None, model=MODEL, tiny_vae=True):
        # the UNet halves the latent twice, so both sides must be multiples of 32
        assert width % 32 == 0 and height % 32 == 0, (width, height)
        from transformers import CLIPTokenizer
        self.w, self.h, self.steps, self.tiny = width, height, steps, tiny_vae
        core = ov.Core()
        cfg = {'PERFORMANCE_HINT': 'LATENCY'}
        if threads:
            cfg['INFERENCE_NUM_THREADS'] = int(threads)
        self.tok1 = CLIPTokenizer.from_pretrained(os.path.join(model, 'tokenizer'))
        self.tok2 = CLIPTokenizer.from_pretrained(os.path.join(model, 'tokenizer_2'))

        def load(part, shapes):
            m = core.read_model(os.path.join(model, part, 'openvino_model.xml'))
            m.reshape(shapes)                      # static shapes compile to far faster kernels
            return core.compile_model(m, 'CPU', cfg)

        lh, lw = height // 8, width // 8
        self.te1 = load('text_encoder', {'input_ids': [1, 77]})
        self.te2 = load('text_encoder_2', {'input_ids': [1, 77]})
        self.unet = load('unet', {'sample': [1, 4, lh, lw], 'timestep': [1],
                                  'encoder_hidden_states': [1, 77, 2048],
                                  'text_embeds': [1, 1280], 'time_ids': [1, 6]})
        if tiny_vae:
            m = core.read_model(TINY_VAE)
            m.reshape({'latent_sample': [1, 4, lh, lw]})
            self.vae = core.compile_model(m, 'CPU', cfg)
        else:
            self.vae = load('vae_decoder', {'latent_sample': [1, 4, lh, lw]})

        full = _sigmas()
        n = steps
        self.timesteps = (np.round(np.arange(1000, 0, -1000 / n)).astype(np.int64) - 1)
        self.sig = np.append(np.interp(self.timesteps, np.arange(1000), full), 0.0)
        self.time_ids = np.array([[height, width, 0, 0, height, width]], np.float32)

    def tokens(self, prompt):
        """How many CLIP tokens a prompt uses; anything past 77 is silently dropped."""
        return len(self.tok1(prompt).input_ids)

    def encode(self, prompt):
        ids1 = self.tok1(prompt, padding='max_length', max_length=77, truncation=True,
                         return_tensors='np').input_ids.astype(np.int64)
        ids2 = self.tok2(prompt, padding='max_length', max_length=77, truncation=True,
                         return_tensors='np').input_ids.astype(np.int64)
        o1 = self.te1({'input_ids': ids1})
        o2 = self.te2({'input_ids': ids2})
        # SDXL conditions on each encoder's penultimate layer, and on encoder 2's pooled output
        h1 = o1[self.te1.output('hidden_states.11')]
        h2 = o2[self.te2.output('hidden_states.31')]
        pooled = o2[self.te2.output('text_embeds')]
        return np.concatenate([h1, h2], axis=-1).astype(np.float32), pooled.astype(np.float32)

    def __call__(self, prompt, seed):
        emb, pooled = self.encode(prompt)
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((1, 4, self.h // 8, self.w // 8)).astype(np.float64) * self.sig[0]
        for i, t in enumerate(self.timesteps):
            s, s_next = self.sig[i], self.sig[i + 1]
            inp = (x / np.sqrt(s * s + 1)).astype(np.float32)
            eps = self.unet({'sample': inp, 'timestep': np.array([t], np.int64),
                             'encoder_hidden_states': emb, 'text_embeds': pooled,
                             'time_ids': self.time_ids})[0].astype(np.float64)
            x0 = x - s * eps
            if s_next == 0:
                x = x0
                break
            up = np.sqrt(s_next ** 2 * (s ** 2 - s_next ** 2) / s ** 2)
            down = np.sqrt(s_next ** 2 - up ** 2)
            x = x0 + eps * down + rng.standard_normal(x.shape) * up
        # the tiny VAE takes the latent as sampled; the full one expects it unscaled first
        lat = x if self.tiny else x / VAE_SCALE
        img = self.vae({'latent_sample': lat.astype(np.float32)})[0][0]
        img = np.clip(img.transpose(1, 2, 0) / 2 + 0.5, 0, 1)
        return Image.fromarray((img * 255 + 0.5).astype(np.uint8))


class Upscaler:
    """Real-ESRGAN general x4v3 on OpenVINO: 768x432 in, 3072x1728 out, which is then taken
    down to 1920x1080 -- so the delivered frame is supersampled rather than stretched."""

    def __init__(self, width, height, threads=None):
        core = ov.Core()
        m = core.read_model(SR_MODEL)
        m.reshape({'input': [1, 3, height, width]})
        cfg = {'PERFORMANCE_HINT': 'LATENCY'}
        if threads:
            cfg['INFERENCE_NUM_THREADS'] = int(threads)
        self.net = core.compile_model(m, 'CPU', cfg)

    def __call__(self, img, size=(1920, 1080)):
        a = np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.net({'input': a})[0][0]
        out = np.clip(out.transpose(1, 2, 0), 0, 1)
        big = Image.fromarray((out * 255 + 0.5).astype(np.uint8))
        return big.resize(size, Image.LANCZOS)


if __name__ == '__main__':
    import sys
    import time
    w, h, steps = (int(a) for a in (sys.argv[1:4] if len(sys.argv) >= 4 else (1024, 576, 1)))
    t0 = time.time()
    g = SDXLTurbo(w, h, steps)
    print(f'compiled {w}x{h} steps={steps} in {time.time()-t0:.1f}s', flush=True)
    p = ('exterior of a tired four-storey brick research building in an industrial park at night '
         'in heavy rain, sodium streetlights, puddles, empty parking lot, cinematic film still, '
         'dramatic lighting, 35mm, rich shadows')
    for k in range(2):
        t0 = time.time()
        im = g(p, 1234 + k)
        print(f'image {k}: {time.time()-t0:.1f}s', flush=True)
    im.save(f'/tmp/claude-0/-home-user-e/b62efde1-8f1a-59c0-a070-04b692eaca77/scratchpad/bench_{w}x{h}_{steps}.png')
