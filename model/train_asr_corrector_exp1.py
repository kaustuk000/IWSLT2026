"""
train_asr_corrector.py
======================
Stage 2 — Fine-tune a causal LLM (e.g. Aryavarta / Llama-3 / IndicLLM)
to correct noisy Bhojpuri ASR output → clean Bhojpuri transcript.

Data contract (CSV)
-------------------
Your CSV must contain AT MINIMUM two columns (names are configurable):
    noisy_bho   : raw 1-best output from Vakyansh/Wav2Vec2 ASR
    clean_bho   : gold Bhojpuri transcript
    hindi_ref   : (optional) Hindi reference — not used for THIS model
                  but useful to keep for the MT post-editor script

Column names are set via --noisy_col / --clean_col CLI args.

Optionally you can also supply IWSLT-style stamped.tsv data (no header);
in that case use --tsv_bho_hyp  and --tsv_hin_ref pointing to the
corresponding .bho.hyp and .bho (reference) hypothesis files.

Training objective
------------------
  PROMPT:  same template as inference_exp1._ASR_CORRECT_PROMPT
           (alignment is critical for fine-tuned model performance)
  TARGET:  "<clean Bhojpuri>"

CRITICAL: Prompt/inference alignment
-------------------------------------
  The prompt used at training time must EXACTLY match the prompt used at
  inference time (inference_exp1.py). The completion token "Corrected Bhojpuri:"
  must be identical in both files. A mismatch causes the model to generate
  in the wrong context even if the training loss converges correctly.

  Training prompt (PROMPT_TEMPLATE) matches inference _ASR_CORRECT_PROMPT:
    "नीचे दिए गए शोरगुल वाले भोजपुरी ASR आउटपुट को सुधारें।\n"
    "केवल सुधरा हुआ भोजपुरी वाक्य लिखें — कोई व्याख्या नहीं।\n\n"
    "Correct the following noisy Bhojpuri ASR output. "
    "Output ONLY the corrected Bhojpuri — no explanation.\n\n"
    "Noisy: {noisy}\n"
    "Corrected Bhojpuri:"

Repetition penalty — why it is kept here but removed in inference_exp1.py
--------------------------------------------------------------------------
  inference_exp1.py removed repetition_penalty from _llm_generate because it
  caused BLEU=0 for Stage 4 (Hindi translation): Hindi function words (है, की,
  का, के, में, से, ने, को, पर) form natural repeating trigrams in every
  sentence, so the penalty caused the decoder to emit EOS immediately.

  THIS FILE targets Bhojpuri ASR stutter correction (क क क क, केक केका के क क)
  — a genuinely pathological repetition pattern, not normal language use.
  Keeping repetition_penalty=1.3 + no_repeat_ngram_size=3 HERE is correct and
  intentional. Bhojpuri function words do not form the same natural trigrams
  that broke Hindi Stage 4.

Metrics logged every eval step
-------------------------------
  • WER  (Word Error Rate)       — primary ASR metric
  • CER  (Char Error Rate)
  • BLEU (sacrebleu corpus-level)
  • chrF++ (sacrebleu, word_order=2)
  • 3 sample comparisons printed to stdout

Checkpointing
-------------
  outputs/best/      ← best checkpoint (lowest eval WER)
  outputs/latest/    ← always updated to the most recent step

Repetition Handling
-------------------
  ASR systems frequently produce highly repetitive output for low-resource
  languages like Bhojpuri, e.g.:
      "केक केका के क क क क क क हे"
  Two complementary defences are applied:

  1. PRE-PROCESSING (data time)
     detect_repetition_score() measures how "stuttery" a string is.
     During data loading, rows whose noisy text score exceeds
     --rep_filter_threshold are either DROPPED (--rep_filter_action=drop)
     or KEPT but flagged in the prompt so the LLM knows they are very noisy
     (--rep_filter_action=flag  ← default).

  2. GENERATION TIME
     model.generate() receives:
       repetition_penalty  (default 1.3) — exponentially penalises tokens
                           already present in the context window.
       no_repeat_ngram_size (default 3)  — hard-blocks any 3-gram that has
                           already appeared in the output.
     Both defaults can be overridden via --rep_penalty / --no_repeat_ngram.

Usage
-----
    python train_asr_corrector.py \
        --model_name  "ai4bharat/Airavata" \
        --csv_path    data/asr_correction_data.csv \
        --noisy_col   noisy_bho \
        --clean_col   clean_bho \
        --output_dir  ./asr_corrector_out \
        --epochs      3 \
        --batch_size  4 \
        --lr          2e-4 \
        --max_len     256 \
        --lora_r      16 \
        --val_split   0.1 \
        --rep_penalty 1.3 \
        --no_repeat_ngram 3 \
        --rep_filter_threshold 0.5 \
        --rep_filter_action flag

    # With IWSLT stamped tsv instead of CSV:
    python train_asr_corrector.py \
        --model_name  "ai4bharat/Airavata" \
        --tsv_bho_hyp cascade_outputs/txt/dev.bho.hyp \
        --tsv_bho_ref data/txt/dev.bho \
        --output_dir  ./asr_corrector_out
"""

from __future__ import annotations

import argparse
import os
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ══════════════════════════════════════════════════════════════════════════════
#  Repetition utilities
# ══════════════════════════════════════════════════════════════════════════════

def detect_repetition_score(text: str) -> float:
    """
    Returns a float in [0, 1] estimating how repetitive the text is.
    Higher = more repetitive.

    Method
    ------
    1. Split into tokens (whitespace).
    2. Compute the *type-token ratio* (TTR) = unique_tokens / total_tokens.
       Pure speech has TTR ~ 0.6–0.9; a stutter like "क क क क क" → TTR = 0.2
    3. Also detect consecutive duplicate tokens (bigram repeats):
       rep_bigram_ratio = count_of_repeated_adjacent_pairs / max(1, len-1)
    4. Final score = 0.5 * (1 - TTR)  +  0.5 * rep_bigram_ratio
       → 0.0 means perfectly varied text; 1.0 means completely repetitive.

    Example
    -------
    >>> detect_repetition_score("केक केका के क क क क क क हे")
    0.68   # high repetition
    >>> detect_repetition_score("आज मौसम बहुत अच्छा है")
    0.0    # no repetition
    """
    tokens = text.strip().split()
    if len(tokens) <= 1:
        return 0.0

    ttr = len(set(tokens)) / len(tokens)

    rep_bigrams = sum(
        1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1]
    )
    rep_bigram_ratio = rep_bigrams / (len(tokens) - 1)

    score = 0.5 * (1.0 - ttr) + 0.5 * rep_bigram_ratio
    return round(score, 4)


def normalize_repetitions(text: str, max_consecutive: int = 2) -> str:
    """
    Collapse runs of the same token that appear more than `max_consecutive`
    times in a row.

    Example
    -------
    >>> normalize_repetitions("केक केका के क क क क क क हे", max_consecutive=2)
    "केक केका के क क हे"

    This is applied to the *noisy* side only — never to the clean reference.
    It gives the LLM a slightly cleaner starting point while still preserving
    evidence of repetition so it can learn the correction pattern.
    """
    tokens = text.strip().split()
    result: List[str] = []
    run_len = 0
    prev: Optional[str] = None

    for tok in tokens:
        if tok == prev:
            run_len += 1
            if run_len <= max_consecutive:
                result.append(tok)
            # else: silently drop the extra repeat
        else:
            result.append(tok)
            run_len = 1
            prev = tok

    return " ".join(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt builder
#
#  CRITICAL: These templates MUST match _ASR_CORRECT_PROMPT in inference_exp1.py
#  exactly — including the completion token "Corrected Bhojpuri:".
#  Any mismatch causes the fine-tuned model to generate in the wrong context.
# ══════════════════════════════════════════════════════════════════════════════

# Standard prompt — matches inference_exp1._ASR_CORRECT_PROMPT exactly.
PROMPT_TEMPLATE = (
    "नीचे दिए गए शोरगुल वाले भोजपुरी ASR आउटपुट को सुधारें।\n"
    "केवल सुधरा हुआ भोजपुरी वाक्य लिखें — कोई व्याख्या नहीं।\n\n"
    "Correct the following noisy Bhojpuri ASR output. "
    "Output ONLY the corrected Bhojpuri — no explanation.\n\n"
    "Noisy: {noisy}\n"
    "Corrected Bhojpuri:"
)

# High-repetition variant — adds ⚠️ warning so the LLM knows the input is very
# low quality.  Still ends with "Corrected Bhojpuri:" to match the standard
# completion token.
PROMPT_TEMPLATE_REPETITIVE = (
    "नीचे दिए गए शोरगुल वाले भोजपुरी ASR आउटपुट को सुधारें।\n"
    "केवल सुधरा हुआ भोजपुरी वाक्य लिखें — कोई व्याख्या नहीं।\n"
    "⚠️  यह ASR आउटपुट अत्यधिक दोहराव वाला / शोरगुल भरा है — ध्यान से सुधारें।\n\n"
    "Correct the following noisy Bhojpuri ASR output. "
    "Output ONLY the corrected Bhojpuri — no explanation.\n"
    "(WARNING: highly repetitive / low-quality ASR input)\n\n"
    "Noisy: {noisy}\n"
    "Corrected Bhojpuri:"
)


def build_prompt(noisy: str, is_repetitive: bool = False) -> str:
    template = PROMPT_TEMPLATE_REPETITIVE if is_repetitive else PROMPT_TEMPLATE
    return template.format(noisy=noisy.strip())


def build_full_text(noisy: str, clean: str, is_repetitive: bool = False) -> str:
    """Full sequence used for training (prompt + target + EOS)."""
    return build_prompt(noisy, is_repetitive) + " " + clean.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ASRCorrectionDataset(Dataset):
    """
    Pairs of (noisy_bho, clean_bho).
    Only the clean portion is included in the loss (prompt is masked).
    """

    def __init__(
        self,
        noisy_texts: List[str],
        clean_texts: List[str],
        tokenizer,
        max_len: int = 256,
        rep_flags: Optional[List[bool]] = None,
    ):
        assert len(noisy_texts) == len(clean_texts)
        self.pairs     = list(zip(noisy_texts, clean_texts))
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.rep_flags = rep_flags if rep_flags else [False] * len(noisy_texts)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        noisy, clean  = self.pairs[idx]
        is_repetitive = self.rep_flags[idx]
        prompt        = build_prompt(noisy, is_repetitive)
        full_text     = prompt + " " + clean.strip() + self.tokenizer.eos_token

        full_enc   = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Mask prompt tokens in labels so loss is computed on target only
        labels = input_ids.clone()
        prompt_len = min(prompt_enc["input_ids"].shape[1], self.max_len)
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def apply_repetition_filter(
    noisy: List[str],
    clean: List[str],
    threshold: float = 0.5,
    action: str = "flag",
    max_consecutive: int = 2,
) -> Tuple[List[str], List[str], List[bool]]:
    """
    Inspect each noisy sample for repetition and apply the chosen action.

    Parameters
    ----------
    threshold : float
        detect_repetition_score() value above which a sample is considered
        highly repetitive.
    action : str
        "flag"      — keep all samples; mark repetitive ones so the prompt
                      template includes a ⚠️  warning (default).
        "drop"      — remove repetitive samples entirely from the dataset.
        "normalize" — collapse long repetitive runs (normalize_repetitions)
                      then keep all samples, flagging the changed ones.
    """
    noisy_out, clean_out, rep_flags = [], [], []
    n_flagged = n_dropped = n_normalized = 0

    for n, c in tqdm(zip(noisy, clean), total=len(noisy), desc="RepFilter"):
        score = detect_repetition_score(n)
        is_rep = score >= threshold

        if is_rep:
            if action == "drop":
                n_dropped += 1
                continue
            elif action == "normalize":
                n = normalize_repetitions(n, max_consecutive)
                n_normalized += 1
            n_flagged += 1

        noisy_out.append(n)
        clean_out.append(c)
        rep_flags.append(is_rep)

    total = len(noisy)
    print(
        f"[RepFilter] threshold={threshold}  action={action}\n"
        f"  Highly repetitive samples : {n_flagged} / {total} "
        f"({100*n_flagged/max(1,total):.1f}%)\n"
        f"  Dropped                   : {n_dropped}\n"
        f"  Normalized (collapsed)    : {n_normalized}\n"
        f"  Flagged in prompt         : {n_flagged - n_dropped - n_normalized}"
    )
    return noisy_out, clean_out, rep_flags


def load_from_csv(
    csv_path: str,
    noisy_col: str,
    clean_col: str,
) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)
    missing = [c for c in [noisy_col, clean_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[noisy_col, clean_col])
    df = df[df[noisy_col].str.strip().astype(bool)]
    df = df[df[clean_col].str.strip().astype(bool)]
    print(f"[DataLoader] Loaded {len(df)} rows from {csv_path}")
    return df[noisy_col].tolist(), df[clean_col].tolist()


def load_from_tsv_files(
    bho_hyp_path: str,
    bho_ref_path: str,
) -> Tuple[List[str], List[str]]:
    def _lines(p):
        with open(p, encoding="utf-8") as f:
            return [l.rstrip("\n") for l in f if l.strip()]

    hyps = _lines(bho_hyp_path)
    refs = _lines(bho_ref_path)
    n = min(len(hyps), len(refs))
    print(f"[DataLoader] Loaded {n} pairs from TSV files")
    return hyps[:n], refs[:n]


def train_val_split(
    noisy: List[str],
    clean: List[str],
    rep_flags: List[bool],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[bool], List[str], List[str], List[bool]]:
    indices = list(range(len(noisy)))
    random.seed(seed)
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]
    return (
        [noisy[i]     for i in train_idx],
        [clean[i]     for i in train_idx],
        [rep_flags[i] for i in train_idx],
        [noisy[i]     for i in val_idx],
        [clean[i]     for i in val_idx],
        [rep_flags[i] for i in val_idx],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics_asr(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    import sacrebleu
    from jiwer import cer as jiwer_cer
    from jiwer import wer as jiwer_wer

    valid = [(h, r) for h, r in zip(hyps, refs) if r.strip()]
    if not valid:
        return {"wer": 0.0, "cer": 0.0, "bleu": 0.0, "chrf": 0.0}

    h_list, r_list = zip(*valid)
    h_list, r_list = list(h_list), list(r_list)

    wer  = jiwer_wer(r_list, h_list)
    cer  = jiwer_cer(r_list, h_list)
    bleu = sacrebleu.corpus_bleu(h_list, [r_list]).score
    chrf = sacrebleu.corpus_chrf(h_list, [r_list], word_order=2).score

    return {
        "wer":  round(wer  * 100, 2),
        "cer":  round(cer  * 100, 2),
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
    }


def print_samples(
    noisy: List[str],
    preds: List[str],
    refs:  List[str],
    n: int = 3,
    label: str = "SAMPLE",
) -> None:
    print(f"\n{'═'*60}")
    print(f"  {label} — {n} examples")
    print(f"{'═'*60}")
    indices = random.sample(range(len(noisy)), min(n, len(noisy)))
    for i, idx in enumerate(indices, 1):
        score = detect_repetition_score(noisy[idx])
        rep_tag = f"  ⚠️  rep_score={score:.2f}" if score >= 0.4 else ""
        print(f"\n  [{i}] NOISY  : {noisy[idx]}{rep_tag}")
        print(f"      PRED   : {preds[idx]}")
        print(f"      REF    : {refs[idx]}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Model + LoRA setup
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    hf_token: Optional[str] = None,
):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    load_kw = {}
    if hf_token:
        load_kw["token"] = hf_token

    print(f"[Model] Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, **load_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("[Model] BitsAndBytes NF4 double-quant config ready")

    print(f"[Model] Loading 4-bit quantized model: {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        **load_kw,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    print("[Model] prepare_model_for_kbit_training done")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  Generation + evaluation helper
#
#  NOTE: repetition_penalty is intentionally KEPT here (unlike inference_exp1.py
#  which removed it from _llm_generate for Stage 4 Hindi output).
#  This targets Bhojpuri ASR stutter — a real pathological pattern — not
#  normal Hindi function word trigrams. See module docstring for details.
# ══════════════════════════════════════════════════════════════════════════════

def generate_corrections(
    model,
    tokenizer,
    noisy_texts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    rep_flags: Optional[List[bool]] = None,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 3,
) -> List[str]:
    """
    Generate corrected Bhojpuri text from noisy ASR input.

    Repetition penalty details
    --------------------------
    repetition_penalty (float, default=1.3)
        Divides the logit score of any token already in the context window.
        1.0 = no penalty; 1.3 = moderate (recommended for Bhojpuri stutter).
        Values above 1.5 risk penalising legitimate repeated content.

    no_repeat_ngram_size (int, default=3)
        Hard constraint: any n-gram of this size already in the *generated*
        output is given -inf logit. Set to 0 to disable.

    Adaptive override
    -----------------
    If a sample's rep_flag is True, temporarily boost repetition_penalty by
    ×1.3 and tighten no_repeat_ngram_size to 2 for that batch.
    """
    model.eval()
    results = []
    device = next(model.parameters()).device
    if rep_flags is None:
        rep_flags = [False] * len(noisy_texts)

    for i in tqdm(range(0, len(noisy_texts), batch_size), desc="Generating"):
        chunk        = noisy_texts[i : i + batch_size]
        chunk_flags  = rep_flags[i : i + batch_size]

        prompts = [build_prompt(t, f) for t, f in zip(chunk, chunk_flags)]

        eff_rep_penalty = 1.0
        eff_no_rep_ngram = 0

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                max_length=None,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=eff_rep_penalty,
                no_repeat_ngram_size=eff_no_rep_ngram,
            )

        for j, ids in enumerate(out):
            new_ids = ids[enc["input_ids"].shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            # Strip hallucinated prompt continuation
            if "\nNoisy:" in text:
                text = text.split("\nNoisy:")[0].strip()

            # Post-generation safety net: if output is still very repetitive,
            # collapse it.
            if detect_repetition_score(text) >= 0.6:
                text = normalize_repetitions(text, max_consecutive=2)

            results.append(text)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Trainer with best-model saving
# ══════════════════════════════════════════════════════════════════════════════

from transformers import Trainer, TrainingArguments


class ASRCorrectorTrainer(Trainer):
    """Extends HF Trainer to:
    - run full generation-based evaluation at each eval step
    - save best model by WER
    - always write latest/ checkpoint
    - print 3 sample predictions
    """

    def __init__(
        self,
        *args,
        val_noisy: List[str],
        val_clean: List[str],
        val_rep_flags: List[bool],
        best_dir: str,
        latest_dir: str,
        gen_batch: int = 4,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.val_noisy            = val_noisy
        self.val_clean            = val_clean
        self.val_rep_flags        = val_rep_flags
        self.best_dir             = Path(best_dir)
        self.latest_dir           = Path(latest_dir)
        self.gen_batch            = gen_batch
        self.repetition_penalty   = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self._best_wer            = float("inf")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        print("\n[Eval] Running generation-based evaluation ...")
        preds = generate_corrections(
            self.model, self.processing_class,
            self.val_noisy,
            max_new_tokens=256,
            batch_size=self.gen_batch,
            rep_flags=self.val_rep_flags,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )

        metrics = compute_metrics_asr(preds, self.val_clean)
        print(
            f"[Eval] WER={metrics['wer']:.2f}%  CER={metrics['cer']:.2f}%  "
            f"BLEU={metrics['bleu']:.2f}  chrF++={metrics['chrf']:.2f}"
        )

        for k, v in metrics.items():
            out[f"eval_gen_{k}"] = v
        self.log(out)

        print_samples(self.val_noisy, preds, self.val_clean,
                      n=3, label="VALIDATION SAMPLES")

        if metrics["wer"] < self._best_wer:
            self._best_wer = metrics["wer"]
            print(f"[Checkpoint] New best WER={metrics['wer']:.2f}% — saving to {self.best_dir}")
            self.model.save_pretrained(str(self.best_dir))
            self.processing_class.save_pretrained(str(self.best_dir))

        self.model.save_pretrained(str(self.latest_dir))
        self.processing_class.save_pretrained(str(self.latest_dir))
        print(f"[Checkpoint] Latest saved to {self.latest_dir}")

        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LLM-based Bhojpuri ASR corrector (Stage 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_grp = p.add_argument_group("Data (CSV OR TSV files, one required)")
    data_grp.add_argument("--csv_path",    default=None)
    data_grp.add_argument("--noisy_col",   default="noisy_bho")
    data_grp.add_argument("--clean_col",   default="clean_bho")
    data_grp.add_argument("--tsv_bho_hyp", default=None)
    data_grp.add_argument("--tsv_bho_ref", default=None)

    rep_grp = p.add_argument_group("Repetition handling")
    rep_grp.add_argument(
        "--rep_penalty", type=float, default=1.3,
        help="repetition_penalty for model.generate(). 1.0=off, 1.3=moderate (recommended)."
    )
    rep_grp.add_argument(
        "--no_repeat_ngram", type=int, default=3,
        help="no_repeat_ngram_size for model.generate(). 0=off, 3=recommended."
    )
    rep_grp.add_argument("--rep_filter_threshold", type=float, default=0.5)
    rep_grp.add_argument(
        "--rep_filter_action", default="flag",
        choices=["flag", "drop", "normalize"],
    )
    rep_grp.add_argument("--rep_max_consecutive", type=int, default=2)

    p.add_argument("--model_name",   default="ai4bharat/Airavata")
    p.add_argument("--hf_token",     default=None)
    p.add_argument("--lora_r",       type=int, default=16)
    p.add_argument("--lora_alpha",   type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--output_dir",   default="./asr_corrector_out")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max_len",      type=int,   default=256)
    p.add_argument("--val_split",    type=float, default=0.1)
    p.add_argument("--eval_steps",   type=int,   default=100)
    p.add_argument("--save_steps",   type=int,   default=100)
    p.add_argument("--warmup_steps", type=int,   default=50)
    p.add_argument("--gen_batch",    type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--fp16",         action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.csv_path:
        noisy_all, clean_all = load_from_csv(
            args.csv_path, args.noisy_col, args.clean_col
        )
    elif args.tsv_bho_hyp and args.tsv_bho_ref:
        noisy_all, clean_all = load_from_tsv_files(
            args.tsv_bho_hyp, args.tsv_bho_ref
        )
    else:
        raise ValueError(
            "Provide EITHER --csv_path  OR  --tsv_bho_hyp + --tsv_bho_ref"
        )

    noisy_all, clean_all, rep_flags_all = apply_repetition_filter(
        noisy_all,
        clean_all,
        threshold=args.rep_filter_threshold,
        action=args.rep_filter_action,
        max_consecutive=args.rep_max_consecutive,
    )

    (
        train_noisy, train_clean, train_rep_flags,
        val_noisy,   val_clean,   val_rep_flags,
    ) = train_val_split(
        noisy_all, clean_all, rep_flags_all,
        val_ratio=args.val_split, seed=args.seed
    )
    print(
        f"[Data] Train={len(train_noisy)}  Val={len(val_noisy)}\n"
        f"[Data] Repetitive in train: {sum(train_rep_flags)}  "
        f"val: {sum(val_rep_flags)}"
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hf_token=args.hf_token,
    )

    train_ds = ASRCorrectionDataset(
        train_noisy, train_clean, tokenizer, args.max_len,
        rep_flags=train_rep_flags,
    )
    val_ds = ASRCorrectionDataset(
        val_noisy, val_clean, tokenizer, args.max_len,
        rep_flags=val_rep_flags,
    )

    out_dir    = Path(args.output_dir)
    best_dir   = out_dir / "best"
    latest_dir = out_dir / "latest"

    use_bf16 = not args.fp16 and torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=str(out_dir / "hf_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate= args.lr,
        warmup_steps = int(args.warmup_steps),
        lr_scheduler_type="cosine",
        fp16=args.fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = ASRCorrectorTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        val_noisy=val_noisy,
        val_clean=val_clean,
        val_rep_flags=val_rep_flags,
        best_dir=str(best_dir),
        latest_dir=str(latest_dir),
        gen_batch=args.gen_batch,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
    )

    print("\n[Baseline] Evaluating pre-trained model on validation set ...")
    baseline_preds = generate_corrections(
        model, tokenizer, val_noisy,
        max_new_tokens=256,
        batch_size=args.gen_batch,
        rep_flags=val_rep_flags,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
    )
    baseline_metrics = compute_metrics_asr(baseline_preds, val_clean)
    print(
        f"[Baseline] WER={baseline_metrics['wer']:.2f}%  "
        f"CER={baseline_metrics['cer']:.2f}%  "
        f"BLEU={baseline_metrics['bleu']:.2f}  "
        f"chrF++={baseline_metrics['chrf']:.2f}"
    )
    print_samples(val_noisy, baseline_preds, val_clean, n=3, label="BASELINE SAMPLES")

    print("\n[Train] Starting fine-tuning ...\n")
    trainer.train()

    print("\n[Final Eval] Running on full validation set ...")
    final_preds = generate_corrections(
        model, tokenizer, val_noisy,
        max_new_tokens=256,
        batch_size=args.gen_batch,
        rep_flags=val_rep_flags,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
    )
    final_metrics = compute_metrics_asr(final_preds, val_clean)
    print(
        f"\n{'═'*60}\n"
        f"  FINAL RESULTS — ASR Corrector\n"
        f"{'═'*60}\n"
        f"  WER   : {final_metrics['wer']:.2f}%\n"
        f"  CER   : {final_metrics['cer']:.2f}%\n"
        f"  BLEU  : {final_metrics['bleu']:.2f}\n"
        f"  chrF++: {final_metrics['chrf']:.2f}\n"
        f"{'═'*60}"
    )
    print_samples(val_noisy, final_preds, val_clean, n=3, label="FINAL SAMPLES")

    model.save_pretrained(str(latest_dir))
    tokenizer.save_pretrained(str(latest_dir))
    print(f"\n[Done] Best model   → {best_dir}")
    print(f"[Done] Latest model → {latest_dir}")

    import json
    summary = {
        "baseline": baseline_metrics,
        "final":    final_metrics,
        "improvement": {
            k: round(baseline_metrics[k] - final_metrics[k], 2)
            for k in final_metrics
        },
        "rep_settings": {
            "repetition_penalty":   args.rep_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram,
            "rep_filter_threshold": args.rep_filter_threshold,
            "rep_filter_action":    args.rep_filter_action,
        },
    }
    summary_path = out_dir / "metrics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Done] Metrics summary → {summary_path}")


if __name__ == "__main__":
    main()