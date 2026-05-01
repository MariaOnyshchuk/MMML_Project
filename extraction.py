"""
extract_attention.py

Extracts per-layer, head-averaged, per-step cross-attention from the Fairseq
DE-EN hallucination model and saves results aligned to the annotated corpus.

Output: a .pkl file with one row per sentence containing:
  - src, mt          : source and translation strings
  - <label columns>  : all hallucination labels from annotated_corpus.csv
  - attn             : np.ndarray [T, n_layers, src_len]
                       head-averaged cross-attention per decoding step per layer
  - attn_aggregated  : np.ndarray [src_len]
                       last-layer, step-averaged distribution (π_M(x) from
                       Guerreiro et al., used by the OT paper as the aggregate)
  - mt_len           : int — number of decoded tokens T

Usage:
  python extract_attention.py \
      --data-dir   path/to/binarized/data \
      --checkpoint path/to/checkpoint_best.pt \
      --corpus     path/to/annotated_corpus.csv \
      --spm-model  path/to/sentencepiece.model \
      --output     attention_per_layer.pkl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ── PyTorch 2.6 compatibility ──────────────────────────────────────────────────
# Fairseq checkpoints contain argparse.Namespace objects, which are rejected by
# torch.load's new weights_only=True default. Patch before Fairseq is imported.
_orig_torch_load = torch.load

def _patched_torch_load(f, map_location=None, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(f, map_location=map_location, **kwargs)

torch.load = _patched_torch_load

from fairseq import checkpoint_utils  # noqa: E402  (must follow the patch)


class CrossAttentionHook:
    """
    Forward hooks on every decoder layer's encoder_attn module.

    Captures the attention tensor at each decoding step. Handles both the
    standard Fairseq shape [B, tgt_pos, src_len] (heads pre-averaged) and the
    [B, heads, tgt_pos, src_len] variant (need_head_weights=True).
    """

    def __init__(self, model):
        self.layer_attentions: dict[int, torch.Tensor] = {}
        self._handles = []
        self._register(model)

    def _register(self, model) -> None:
        for layer_idx, layer in enumerate(model.decoder.layers):
            def _make_hook(idx: int):
                def _hook(module, input, output):
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self.layer_attentions[idx] = attn_weights.detach().cpu()
                return _hook

            handle = layer.encoder_attn.register_forward_hook(_make_hook(layer_idx))
            self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def current_step_vectors(self) -> list[np.ndarray]:
        """Head-averaged, last-position [src_len] vector per layer, in layer order."""
        if not self.layer_attentions:
            return []

        per_layer = []
        for l in sorted(self.layer_attentions):
            a = self.layer_attentions[l]
            if a.dim() == 4:
                a = a[0].mean(dim=0)[-1]
            elif a.dim() == 3:
                a = a[0, -1]
            else:
                a = a.view(-1)

            a = a.numpy().astype(np.float64)
            a = np.clip(a, 0.0, None)
            total = a.sum()
            if total > 0:
                a /= total
            per_layer.append(a)

        return per_layer


def load_spm(spm_path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_path))
    return sp


def extract(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model …")
    models, _, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.checkpoint],
        arg_overrides={"data": args.data_dir},
    )
    model = models[0].eval().to(device)

    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    print("Loading annotated corpus …")
    corpus = pd.read_csv(args.corpus, engine="python", quoting=0, on_bad_lines="warn")

    sp = load_spm(args.spm_model)
    hook = CrossAttentionHook(model)
    results = []

    print(f"Extracting attention for {len(corpus)} sentences …")
    for _, row in tqdm(corpus.iterrows(), total=len(corpus)):
        src_ids = [src_dict.index(p) for p in sp.EncodeAsPieces(str(row["src"]))] + [src_dict.eos()]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(device)

        step_attentions: list[np.ndarray] = []

        with torch.no_grad():
            encoder_out = model.encoder(src_tensor, src_lengths=src_lengths)

            max_len = min(int(src_tensor.size(1) * 1.5) + 10, 200)
            prev_tokens = torch.full((1, 1), tgt_dict.eos(), dtype=torch.long, device=device)

            for _ in range(max_len):
                hook.layer_attentions.clear()

                decoder_out = model.decoder(prev_tokens, encoder_out=encoder_out, features_only=False)
                next_token = decoder_out[0][:, -1, :].argmax(dim=-1, keepdim=True)

                vectors = hook.current_step_vectors()
                if vectors:
                    step_attentions.append(np.stack(vectors, axis=0))

                prev_tokens = torch.cat([prev_tokens, next_token], dim=1)
                if next_token.item() == tgt_dict.eos():
                    break

        if step_attentions:
            attn_array = np.stack(step_attentions, axis=0)
        else:
            attn_array = np.zeros((1, 6, len(src_ids)), dtype=np.float64)

        last_layer_avg = attn_array[:, -1, :].mean(axis=0)
        total = last_layer_avg.sum()
        if total > 0:
            last_layer_avg /= total

        record = row.to_dict()
        record["attn"]            = attn_array
        record["attn_aggregated"] = last_layer_avg
        record["mt_len"]          = attn_array.shape[0]
        results.append(record)

        if len(results) == 1:
            T, L, S = attn_array.shape
            print(f"\n✓ First sentence — attn shape: ({T}, {L}, {S})  "
                  f"[T={T} steps, {L} layers, src_len={S}]\n")

    hook.remove()

    df = pd.DataFrame(results)
    df.to_pickle(Path(args.output))
    print(f"Saved {len(df)} rows → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-layer cross-attention from a Fairseq DE-EN model."
    )
    parser.add_argument("--data-dir",   required=True,
                        help="Path to binarized Fairseq data directory")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint_best.pt")
    parser.add_argument("--corpus",     required=True,
                        help="Path to annotated_corpus.csv")
    parser.add_argument("--spm-model",  required=True,
                        help="Path to sentencepiece .model file")
    parser.add_argument("--output",     default="attention_per_layer.pkl",
                        help="Output .pkl path (default: attention_per_layer.pkl)")
    args = parser.parse_args()
    extract(args)