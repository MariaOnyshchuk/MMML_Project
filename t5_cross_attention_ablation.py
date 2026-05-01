"""
t5_cross_attention_ablation.py
──────────────────────────────
T5-small cross-attention ablation study on CNN/DailyMail.

Experiments:
  baseline   – all 6 cross-attention layers intact
  individual – zero out one layer at a time  (L1 … L6)
  cumulative – zero out L1, L1+L2, … up to all 6

Run:
  pip install transformers datasets evaluate rouge_score sentencepiece torch
  python t5_cross_attention_ablation.py
"""

import json, copy, warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Block
from datasets import load_dataset
import evaluate

warnings.filterwarnings("ignore")

MODEL_NAME   = "t5-base"
DATASET_NAME = "cnn_dailymail"
DATASET_VER  = "3.0.0"
SPLIT        = "test"
N_SAMPLES    = 100    
MAX_INPUT    = 512
MAX_TARGET   = 128
NUM_BEAMS    = 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_LAYERS     = 12
OUTPUT_JSON  = "ablation_results.json"

def zero_cross_attn(model, layer_idx):
    enc_dec = model.decoder.block[layer_idx].layer[1].EncDecAttention
    for proj in [enc_dec.q, enc_dec.k, enc_dec.v, enc_dec.o]:
        proj.weight.data.zero_()
        if proj.bias is not None:
            proj.bias.data.zero_()


def make_zeroed_model(base, layers):
    m = copy.deepcopy(base)
    for i in layers:
        zero_cross_attn(m, i)
    return m


class T5BlockNoCross(T5Block):
    """T5Block that skips cross-attention in forward (no computation, not just zeros)."""
    def forward(self, hidden_states, attention_mask=None, position_bias=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                encoder_decoder_position_bias=None, layer_head_mask=None,
                cross_attn_layer_head_mask=None, past_key_value=None,
                use_cache=False, output_attentions=False, return_dict=True):

        self_attn_past = past_key_value[:2] if past_key_value is not None else None
        sa_out = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_kv = sa_out[0], sa_out[1]

        

        hidden_states = self.layer[2](hidden_states)  

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_kv,)
        if output_attentions:
            outputs += (None,)
        return outputs


def make_skip_model(base, layers):
    m = copy.deepcopy(base)
    for i in layers:
        old = m.decoder.block[i]
        new = T5BlockNoCross(m.config, has_relative_attention_bias=(i == 0))
        new.load_state_dict(old.state_dict(), strict=False)
        m.decoder.block[i] = new
    return m


def evaluate_model(model, tokenizer, dataset, n, desc=""):
    rouge = evaluate.load("rouge")
    preds, refs = [], []

    model.to(DEVICE).eval()
    for sample in tqdm(dataset.select(range(min(n, len(dataset)))),
                       desc=f"  {desc:<25}", leave=False, ncols=80):
        enc = tokenizer(
            "summarize: " + sample["article"],
            return_tensors="pt", max_length=MAX_INPUT, truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=MAX_TARGET,
                                 num_beams=NUM_BEAMS, early_stopping=True)
        preds.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        refs.append(sample["highlights"])

    raw = rouge.compute(predictions=preds, references=refs)
    return {k: round(v * 100, 4) for k, v in raw.items()}


def main():
    print(f"Device: {DEVICE}")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    ds = load_dataset(DATASET_NAME, DATASET_VER, split=SPLIT, trust_remote_code=True)
    results = {}

    print("\n── Baseline")
    results["baseline"] = {
        "layers_zeroed": [],
        "scores": evaluate_model(base_model, tokenizer, ds, N_SAMPLES, "Baseline"),
    }
    print("  ", results["baseline"]["scores"])

    print("\n── Individual layer ablation")
    results["individual"] = {}
    for i in range(5, N_LAYERS):
        key = f"zero_L{i+1}"
        m   = make_zeroed_model(base_model, [i])
        sc  = evaluate_model(m, tokenizer, ds, N_SAMPLES, f"Zero L{i+1}")
        results["individual"][key] = {"layers_zeroed": [i], "scores": sc}
        print(f"  L{i+1}: {sc}")
        del m

    
    print("\n── Cumulative ablation")
    results["cumulative"] = {}
    for n in range(1, N_LAYERS + 1):
        layers = list(range(n))
        key    = f"zero_L1-L{n}"
        m      = make_zeroed_model(base_model, layers)
        sc     = evaluate_model(m, tokenizer, ds, N_SAMPLES, f"Zero L1-L{n}")
        results["cumulative"][key] = {"layers_zeroed": layers, "scores": sc}
        print(f"  L1-L{n}: {sc}")
        del m

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON}")

    W = 70
    print("\n" + "="*W)
    print(f"{'Experiment':<22} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'R-Lsum':>8}")
    print("="*W)
    def row(name, sc):
        print(f"{name:<22} {sc.get('rouge1',0):>8.2f} {sc.get('rouge2',0):>8.2f} "
              f"{sc.get('rougeL',0):>8.2f} {sc.get('rougeLsum',0):>8.2f}")
    row("Baseline", results["baseline"]["scores"])
    print("-"*W)
    for k,v in results["individual"].items(): row(k, v["scores"])
    print("-"*W)
    for k,v in results["cumulative"].items(): row(k, v["scores"])
    print("="*W)

if __name__ == "__main__":
    main()
