"""Замена vLLM через transformers (видеокарта NVIDIA, если есть, иначе процессор): только то, что вызывает StepCAD fast_infer
(LLM(...).generate([{'prompt_embeds': emb}], SamplingParams))."""
import os, time
import torch
from transformers import AutoModelForCausalLM

class SamplingParams:
    def __init__(self, n=1, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=256,
                 stop_token_ids=None, seed=None, **kw):
        self.n, self.temperature, self.top_p = n, temperature, top_p
        self.repetition_penalty, self.max_tokens = repetition_penalty, max_tokens
        self.stop_token_ids, self.seed = stop_token_ids or [], seed

class _O:
    def __init__(self, ids): self.token_ids = ids
class _R:
    def __init__(self, outs): self.outputs = outs

class LLM:
    def __init__(self, model, tokenizer=None, dtype='bfloat16', seed=0, **kw):
        torch.set_num_threads(int(os.environ.get('SHIM_THREADS', '4')))
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(self.dev).eval()
        self.cap = int(os.environ.get('SHIM_MAX_NEW', '1200'))
        self.batch = int(os.environ.get('SHIM_BATCH', '4'))

    @torch.no_grad()
    def generate(self, prompts, sp, use_tqdm=False):
        res = []
        for p in prompts:
            emb = p['prompt_embeds'].to(self.dev, torch.bfloat16)[None]
            if sp.seed is not None:
                torch.manual_seed(sp.seed)
            t = time.time()
            rows = []
            # по нескольку вариантов за раз: 16 сразу не помещаются в 8 ГБ видеопамяти, и Windows
            # молча уводит память в общую ОЗУ — генерация становится в разы медленнее
            left = sp.n
            while left > 0:
                k = min(left, self.batch)
                out = self.m.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=self.dev),
                                      do_sample=True, temperature=sp.temperature, top_p=sp.top_p,
                                      repetition_penalty=sp.repetition_penalty, max_new_tokens=min(sp.max_tokens, self.cap),
                                      num_return_sequences=k, eos_token_id=sp.stop_token_ids or None,
                                      pad_token_id=(sp.stop_token_ids or [0])[0])
                rows += out.tolist()
                left -= k
            outs = []
            for row in rows:
                ids = []
                for x in row:
                    if x in sp.stop_token_ids: break
                    ids.append(x)
                outs.append(_O(ids))
            print('shim: %d x %d tok in %.0f s' % (sp.n, max(len(o.token_ids) for o in outs), time.time() - t), flush=True)
            res.append(_R(outs))
        return res
