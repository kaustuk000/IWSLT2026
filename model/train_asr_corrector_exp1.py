"""
train_asr_corrector.py
======================
Stage 2 — Fine-tune ai4bharat/Airavata (or compatible causal LLM)
to correct noisy Bhojpuri ASR output → clean Bhojpuri transcript.

Data contract (CSV)
-------------------
Your CSV must contain AT MINIMUM two columns (names are configurable):
    noisy_bho   : raw 1-best output from ASR (e.g. Whisper)
    clean_bho   : gold Bhojpuri transcript

Column names are set via --noisy_col / --clean_col CLI args.

Optionally supply IWSLT-style stamped.tsv data (no header);
use --tsv_bho_hyp and --tsv_bho_ref pointing to the corresponding files.

Training objective
------------------
  PROMPT:  Airavata <|user|>/<|assistant|> chat format (matches HF model card)
  TARGET:  "<clean Bhojpuri>"

CRITICAL: Prompt/inference alignment
-------------------------------------
  The prompt used at training MUST EXACTLY match the prompt used at inference.
  The role tags and instruction text must be identical in both files.

Prompt format (Airavata / Tulu chat template)
---------------------------------------------
  <s><|user|>
  {instruction}
  <|assistant|>
  {target}</s>

  The instruction asks the model to correct the noisy Bhojpuri ASR output.
  The target is the clean Bhojpuri sentence only — no explanation.

Repetition penalty
------------------
  A mild repetition_penalty (default 1.2) is applied at generation time.
  This is a light general-purpose guard. Set to 1.0 to disable entirely.

Metrics logged every eval step
-------------------------------
  WER  (Word Error Rate)       — primary ASR metric
  CER  (Char Error Rate)
  BLEU (sacrebleu corpus-level)
  chrF++ (sacrebleu, word_order=2)
  3 sample comparisons printed to stdout

Checkpointing
-------------
  outputs/best/      <- best checkpoint (lowest eval WER)
  outputs/latest/    <- always updated to the most recent step

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
        --rep_penalty 1.2

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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt builder — Airavata / Tulu chat format
#
#  CRITICAL: This MUST match inference exactly.
#  Role tags: <|user|>, <|assistant|>
#  BOS/EOS tokens are handled by tokenizer or added explicitly in build_full_text.
#
#  The instruction is bilingual (Hindi + English) so the model gets full context
#  regardless of which language direction it was primarily trained on.
# ══════════════════════════════════════════════════════════════════════════════

ASR_CORRECT_INSTRUCTION = (
    "नीचे दिए गए भोजपुरी ASR आउटपुट को सुधारें। "
    "केवल सुधरा हुआ भोजपुरी वाक्य लिखें — कोई अतिरिक्त व्याख्या नहीं।\n\n"
    "Correct the following Bhojpuri ASR output. "
    "Write ONLY the corrected Bhojpuri sentence — no explanation, no commentary.\n\n"
    "Noisy ASR: {noisy}"
)


def build_prompt(noisy: str) -> str:
    """
    Build the <|user|>...<|assistant|> portion of the prompt.
    Does NOT include the leading BOS token — caller adds it when needed.

    Used at inference time:
        prompt = bos + build_prompt(noisy_text)

    Format:
        <|user|>
        {instruction}
        <|assistant|>

    """
    instruction = ASR_CORRECT_INSTRUCTION.format(noisy=noisy.strip())
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


def build_full_text(noisy: str, clean: str, bos: str = "<s>", eos: str = "</s>") -> str:
    """
    Full sequence used for training (BOS + prompt + target + EOS).

    Format:
        <s><|user|>
        {instruction}
        <|assistant|>
        {clean}</s>
    """
    return bos + build_prompt(noisy) + clean.strip() + eos


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class ASRCorrectionDataset(Dataset):
    """
    Pairs of (noisy_bho, clean_bho).
    Only the clean portion is included in the loss (prompt tokens masked to -100).
    """

    def __init__(
        self,
        noisy_texts: List[str],
        clean_texts: List[str],
        tokenizer,
        max_len: int = 256,
    ):
        assert len(noisy_texts) == len(clean_texts)
        self.pairs     = list(zip(noisy_texts, clean_texts))
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.bos       = tokenizer.bos_token or "<s>"
        self.eos       = tokenizer.eos_token or "</s>"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        noisy, clean = self.pairs[idx]

        full_text  = build_full_text(noisy, clean, bos=self.bos, eos=self.eos)
        prompt_str = self.bos + build_prompt(noisy)  # measure prompt length only

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids      = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Mask prompt tokens in labels so loss is computed only on target
        labels = input_ids.clone()
        prompt_len = min(prompt_enc["input_ids"].shape[1], self.max_len)
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100  # also mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

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
    n    = min(len(hyps), len(refs))
    print(f"[DataLoader] Loaded {n} pairs from TSV files")
    return hyps[:n], refs[:n]


def train_val_split(
    noisy: List[str],
    clean: List[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    indices = list(range(len(noisy)))
    random.seed(seed)
    random.shuffle(indices)
    n_val     = max(1, int(len(indices) * val_ratio))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]
    return (
        [noisy[i] for i in train_idx],
        [clean[i] for i in train_idx],
        [noisy[i] for i in val_idx],
        [clean[i] for i in val_idx],
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
    print(f"\n{'='*60}")
    print(f"  {label} — {n} examples")
    print(f"{'='*60}")
    indices = random.sample(range(len(noisy)), min(n, len(noisy)))
    for i, idx in enumerate(indices, 1):
        print(f"\n  [{i}] NOISY : {noisy[idx]}")
        print(f"      PRED  : {preds[idx]}")
        print(f"      REF   : {refs[idx]}")
    print(f"{'='*60}\n")


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
        model_name, use_fast=True, padding_side="left", **load_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
#  Generation helper
# ══════════════════════════════════════════════════════════════════════════════

def generate_corrections(
    model,
    tokenizer,
    noisy_texts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    repetition_penalty: float = 1.2,
) -> List[str]:
    """
    Generate corrected Bhojpuri text from noisy ASR input.

    Uses the Airavata chat format: <s><|user|>...<|assistant|>
    Greedy decoding (do_sample=False) for deterministic output.

    repetition_penalty
        1.0  = off (no penalty)
        1.2  = mild default — light guard against degenerate repetition
        1.3  = stronger, use if you observe heavy output repetition
    """
    model.eval()
    results = []
    device  = next(model.parameters()).device
    bos     = tokenizer.bos_token or "<s>"

    for i in tqdm(range(0, len(noisy_texts), batch_size), desc="Generating"):
        chunk   = noisy_texts[i : i + batch_size]
        prompts = [bos + build_prompt(t) for t in chunk]

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
                max_length=None,          # suppress conflict warning with model's default max_length
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )

        for j, ids in enumerate(out):
            new_ids = ids[enc["input_ids"].shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            # Strip any hallucinated prompt continuation
            for stop in ["\nNoisy ASR:", "<|user|>", "<|assistant|>"]:
                if stop in text:
                    text = text.split(stop)[0].strip()

            results.append(text)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Custom Trainer with best-model saving
# ══════════════════════════════════════════════════════════════════════════════

from transformers import Trainer, TrainingArguments


class ASRCorrectorTrainer(Trainer):
    """
    Extends HF Trainer to:
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
        best_dir: str,
        latest_dir: str,
        gen_batch: int = 4,
        repetition_penalty: float = 1.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.val_noisy          = val_noisy
        self.val_clean          = val_clean
        self.best_dir           = Path(best_dir)
        self.latest_dir         = Path(latest_dir)
        self.gen_batch          = gen_batch
        self.repetition_penalty = repetition_penalty
        self._best_wer          = float("inf")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        out = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        print("\n[Eval] Running generation-based evaluation ...")
        preds = generate_corrections(
            self.model,
            self.processing_class,
            self.val_noisy,
            max_new_tokens=256,
            batch_size=self.gen_batch,
            repetition_penalty=self.repetition_penalty,
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
            print(f"[Checkpoint] New best WER={metrics['wer']:.2f}% -> saving to {self.best_dir}")
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
        description="Train LLM-based Bhojpuri ASR corrector (Stage 2) — Airavata chat format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_grp = p.add_argument_group("Data (CSV OR TSV files, one required)")
    data_grp.add_argument("--csv_path",    default=None)
    data_grp.add_argument("--noisy_col",   default="noisy_bho")
    data_grp.add_argument("--clean_col",   default="clean_bho")
    data_grp.add_argument("--tsv_bho_hyp", default=None)
    data_grp.add_argument("--tsv_bho_ref", default=None)

    p.add_argument(
        "--rep_penalty", type=float, default=1.2,
        help="repetition_penalty for model.generate(). 1.0=off, 1.2=mild default.",
    )

    p.add_argument("--model_name",   default="ai4bharat/Airavata")
    p.add_argument("--hf_token",     default=None)
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
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

    # ── Load data ──────────────────────────────────────────────────────────
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

    train_noisy, train_clean, val_noisy, val_clean = train_val_split(
        noisy_all, clean_all,
        val_ratio=args.val_split,
        seed=args.seed,
    )
    print(f"[Data] Train={len(train_noisy)}  Val={len(val_noisy)}")

    # ── Load model ─────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hf_token=args.hf_token,
    )

    # ── Build datasets ─────────────────────────────────────────────────────
    train_ds = ASRCorrectionDataset(train_noisy, train_clean, tokenizer, args.max_len)
    val_ds   = ASRCorrectionDataset(val_noisy,   val_clean,   tokenizer, args.max_len)

    # ── Training arguments ─────────────────────────────────────────────────
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
        learning_rate=args.lr,
        warmup_steps=int(args.warmup_steps),
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
        best_dir=str(best_dir),
        latest_dir=str(latest_dir),
        gen_batch=args.gen_batch,
        repetition_penalty=args.rep_penalty,
    )

    # ── Baseline evaluation ────────────────────────────────────────────────
    print("\n[Baseline] Evaluating pre-trained model on validation set ...")
    baseline_preds = generate_corrections(
        model, tokenizer, val_noisy,
        max_new_tokens=256,
        batch_size=args.gen_batch,
        repetition_penalty=args.rep_penalty,
    )
    baseline_metrics = compute_metrics_asr(baseline_preds, val_clean)
    print(
        f"[Baseline] WER={baseline_metrics['wer']:.2f}%  "
        f"CER={baseline_metrics['cer']:.2f}%  "
        f"BLEU={baseline_metrics['bleu']:.2f}  "
        f"chrF++={baseline_metrics['chrf']:.2f}"
    )
    print_samples(val_noisy, baseline_preds, val_clean, n=3, label="BASELINE SAMPLES")

    # ── Fine-tuning ────────────────────────────────────────────────────────
    print("\n[Train] Starting fine-tuning ...\n")
    trainer.train()

    # ── Final evaluation ───────────────────────────────────────────────────
    print("\n[Final Eval] Running on full validation set ...")
    final_preds = generate_corrections(
        model, tokenizer, val_noisy,
        max_new_tokens=256,
        batch_size=args.gen_batch,
        repetition_penalty=args.rep_penalty,
    )
    final_metrics = compute_metrics_asr(final_preds, val_clean)
    print(
        f"\n{'='*60}\n"
        f"  FINAL RESULTS — ASR Corrector\n"
        f"{'='*60}\n"
        f"  WER   : {final_metrics['wer']:.2f}%\n"
        f"  CER   : {final_metrics['cer']:.2f}%\n"
        f"  BLEU  : {final_metrics['bleu']:.2f}\n"
        f"  chrF++: {final_metrics['chrf']:.2f}\n"
        f"{'='*60}"
    )
    print_samples(val_noisy, final_preds, val_clean, n=3, label="FINAL SAMPLES")

    # ── Save final model ───────────────────────────────────────────────────
    model.save_pretrained(str(latest_dir))
    tokenizer.save_pretrained(str(latest_dir))
    print(f"\n[Done] Best model   -> {best_dir}")
    print(f"[Done] Latest model -> {latest_dir}")

    # ── Save metrics summary ───────────────────────────────────────────────
    import json
    summary = {
        "baseline":    baseline_metrics,
        "final":       final_metrics,
        "improvement": {
            k: round(baseline_metrics[k] - final_metrics[k], 2)
            for k in final_metrics
        },
        "rep_settings": {
            "repetition_penalty": args.rep_penalty,
        },
    }
    summary_path = out_dir / "metrics_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Done] Metrics summary -> {summary_path}")


if __name__ == "__main__":
    main()