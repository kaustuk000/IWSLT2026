import os
import json
import warnings
import torch
import sys
import subprocess
import argparse
from tqdm.auto import tqdm
import numpy as np
from transformers import NllbTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import gc
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from functools import partial
from sacrebleu.metrics import BLEU, CHRF
import multiprocess.resource_tracker as _rt

# ── Silence noisy warnings ────────────────────────────────────────────────

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
_rt.ResourceTracker.__del__ = lambda self: None
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.data.data_collator")
torch.backends.cudnn.benchmark = True

# ── Argument Parser ───────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune NLLB-200 for Bhojpuri-Hindi machine translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ──────────────────────────────────────────────────────
    parser.add_argument("--train_file",    type=str,   default="data/MT_data/train.csv",
                        help="Path to train CSV. Defaults to <work_dir>/data/MT_data/train.csv")
    parser.add_argument("--dev_file",      type=str,   default="data/MT_data/dev.csv",
                        help="Path to dev CSV.   Defaults to <work_dir>/data/MT_data/dev.csv")
    parser.add_argument("--work_dir",      type=str,   default=os.getcwd(),
                        help="Working / output root directory")

    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument("--model_name",    type=str,   default="facebook/nllb-200-distilled-600M",
                        help="HuggingFace model name or local path to load from")
    parser.add_argument("--src_lang",      type=str,   default="bho_Deva",
                        help="NLLB source language code")
    parser.add_argument("--tgt_lang",      type=str,   default="hin_Deva",
                        help="NLLB target language code")

    # ── Training hyper-parameters ────────────────────────────────────────
    parser.add_argument("--batch_size",        type=int,   default=16)
    parser.add_argument("--max_length",        type=int,   default=128)
    parser.add_argument("--epochs",            type=int,   default=10)
    parser.add_argument("--lr",                type=float, default=1e-4,  help="Learning rate for Adafactor")
    parser.add_argument("--weight_decay",      type=float, default=1e-3)
    parser.add_argument("--clip_threshold",    type=float, default=1.0,   help="Adafactor clip threshold")
    parser.add_argument("--warmup_steps",      type=int,   default=500)
    parser.add_argument("--checkpoint_every",  type=int,   default=1000,  help="Save a checkpoint every N global steps")
    parser.add_argument("--eval_max_batches",  type=int,   default=0,     help="Limit generation-based dev evaluation to N batches; 0 uses the full dev set")

    # ── Output naming ────────────────────────────────────────────────────
    parser.add_argument("--run_name",      type=str,   default="model-mt-v1",
                        help="Base name used for saved model folders")

    # ── Resume ───────────────────────────────────────────────────────────
    parser.add_argument("--resume",        type=str,   default=None,
                        help=(
                            "Path to a checkpoint directory to resume from.  "
                            "The directory must contain model weights AND a "
                            "resume_state.json file written by this script."
                        ))

    return parser.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────

def run(cmd):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr.strip()}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def get_device(model):
    if isinstance(model, torch.nn.DataParallel):
        return next(model.module.parameters()).device
    return next(model.parameters()).device

def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def save_model(
    path,
    model,
    tokenizer,
    global_step,
    epoch,
    optimizer,
    scheduler,
    scaler,
    loss_log,
    best_bleu=None,
    best_chrf=None,
):
    """Save model weights + tokenizer + full resume state."""
    os.makedirs(path, exist_ok=True)
    unwrap(model).save_pretrained(path)
    tokenizer.save_pretrained(path)

    # ── Optimizer / scheduler / scaler state ──────────────────────────
    torch.save(optimizer.state_dict(),  os.path.join(path, "optimizer.pt"))
    torch.save(scheduler.state_dict(),  os.path.join(path, "scheduler.pt"))
    torch.save(scaler.state_dict(),     os.path.join(path, "scaler.pt"))

    # ── Lightweight JSON resume metadata ──────────────────────────────
    state = {
        "global_step": global_step,
        "epoch":       epoch,
        "loss_log":    loss_log,
    }
    if best_bleu is not None:
        state["best_bleu"] = best_bleu
    if best_chrf is not None:
        state["best_chrf"] = best_chrf
    with open(os.path.join(path, "resume_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print(f"Saved → {path}  (step {global_step}, epoch {epoch})")

def flush_step_losses(path, step_loss_buffer):
    if not step_loss_buffer:
        return
    with open(path, "a") as f:
        for step, loss_val in step_loss_buffer:
            f.write(f"{step},{loss_val:.6f}\n")
    step_loss_buffer.clear()

# ── Tokenize function ─────────────────────────────────────────────────────

def tokenize(batch, tokenizer, max_length, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    enc = tokenizer(batch["bho"], padding=False, truncation=True, max_length=max_length)

    tokenizer.src_lang = tgt_lang
    dec = tokenizer(batch["hin"], padding=False, truncation=True, max_length=max_length)
    tokenizer.src_lang = src_lang

    labels = [
        [-100 if t == tokenizer.pad_token_id else t for t in ids]
        for ids in dec["input_ids"]
    ]
    return {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels":         labels,
    }

def evaluate_generation_metrics(model, tokenizer, dev_loader, tgt_lang, device, max_new_tokens, max_batches=0):
    bleu_metric = BLEU()
    chrf_metric = CHRF()
    hypotheses = []
    references = []
    generator = unwrap(model)
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    for batch_idx, batch in enumerate(tqdm(dev_loader, desc="Generation eval", leave=False)):
        if max_batches and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )
        safe_labels = batch["labels"].clone()
        safe_labels[safe_labels == -100] = tokenizer.pad_token_id
        hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        references.extend(tokenizer.batch_decode(safe_labels, skip_special_tokens=True))

    bleu = bleu_metric.corpus_score(hypotheses, [references]).score if hypotheses else 0.0
    chrf = chrf_metric.corpus_score(hypotheses, [references]).score if hypotheses else 0.0
    return bleu, chrf

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Working directory ────────────────────────────────────────────────
    WORK_DIR = args.work_dir
    os.makedirs(WORK_DIR, exist_ok=True)
    os.chdir(WORK_DIR)
    print(f"Working dir : {WORK_DIR}")

    # ── Resolve data paths ───────────────────────────────────────────────
    MT_train = args.train_file or os.path.join(WORK_DIR, "data", "MT_data", "train.csv")
    MT_dev   = args.dev_file   or os.path.join(WORK_DIR, "data", "MT_data", "dev.csv")

    # ── GPU info ─────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name}  |  VRAM: {p.total_memory/1e9:.1f} GB")
        DEVICE = f"cuda:{max(range(n), key=lambda i: torch.cuda.get_device_properties(i).total_memory)}"
    else:
        print("No GPU — running on CPU")
        DEVICE = "cpu"

    print(f"Device : {DEVICE}")
    CUDA = torch.cuda.is_available()
    num_workers = min(4, os.cpu_count() or 1)

    # ── Decide where to load model weights from ───────────────────────────
    # If resuming, load from checkpoint dir; else from HF hub / local path.
    model_load_path = args.resume if args.resume else args.model_name
    print(f"Loading model from : {model_load_path}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = NllbTokenizer.from_pretrained(
        model_load_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    # ── Dataset + DataLoader ─────────────────────────────────────────────
    train_ds = load_dataset("csv", data_files=MT_train, split="train")
    dev_ds   = load_dataset("csv", data_files=MT_dev,   split="train")

    _tokenize = partial(
        tokenize,
        tokenizer=tokenizer,
        max_length=args.max_length,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    train_ds = train_ds.map(_tokenize, batched=True, batch_size=256, num_proc=None)
    dev_ds   = dev_ds.map(_tokenize,   batched=True, batch_size=256, num_proc=None)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev_ds.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=CUDA,
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=CUDA,
        persistent_workers=num_workers > 0,
        collate_fn=collator,
    )

    print(f"Train samples : {len(train_ds):,}")
    print(f"Dev   samples : {len(dev_ds):,}")
    print(f"Train batches : {len(train_loader):,}")
    print(f"Dev   batches : {len(dev_loader):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    cleanup()
    model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path)
    model = model.to(DEVICE)
    print(f"Model device  : {next(model.parameters()).device}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = Adafactor(
        [p for p in unwrap(model).parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=args.lr,
        clip_threshold=args.clip_threshold,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    scheduler   = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    scaler      = GradScaler("cuda", enabled=CUDA)

    # ── Resume state ──────────────────────────────────────────────────────
    global_step    = 0
    start_epoch    = 1
    loss_log       = []
    all_train_losses = []
    best_bleu = float("-inf")
    best_chrf = float("-inf")

    if args.resume:
        state_file = os.path.join(args.resume, "resume_state.json")
        if not os.path.isfile(state_file):
            raise FileNotFoundError(
                f"--resume was set to '{args.resume}' but no resume_state.json found there. "
                "Make sure the checkpoint was saved by this script."
            )
        with open(state_file) as f:
            state = json.load(f)

        global_step = state["global_step"]
        # Resume from the epoch AFTER the last completed one
        start_epoch = state["epoch"] + 1
        loss_log    = state.get("loss_log", [])
        best_bleu   = state.get("best_bleu", float("-inf"))
        best_chrf   = state.get("best_chrf", float("-inf"))

        # Restore optimizer / scheduler / scaler
        opt_path  = os.path.join(args.resume, "optimizer.pt")
        sch_path  = os.path.join(args.resume, "scheduler.pt")
        scl_path  = os.path.join(args.resume, "scaler.pt")

        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE))
            print("Restored optimizer state")
        if os.path.isfile(sch_path):
            scheduler.load_state_dict(torch.load(sch_path, map_location=DEVICE))
            print("Restored scheduler state")
        if os.path.isfile(scl_path) and CUDA:
            scaler.load_state_dict(torch.load(scl_path, map_location=DEVICE))
            print("Restored scaler state")

        print(f"Resuming from step {global_step}, starting at epoch {start_epoch}")

        if start_epoch > args.epochs:
            print(f"All {args.epochs} epoch(s) already completed in the loaded checkpoint. Nothing to do.")
            return

    print(f"Total training steps (this run) : {total_steps:,}")

    # ── Save paths ────────────────────────────────────────────────────────
    MODEL_BASE_PATH = os.path.join(WORK_DIR, "bho_hin_mt", args.run_name)
    CHECKPOINT_PATH = os.path.join(WORK_DIR, "bho_hin_mt", "checkpoints")
    LOSS_LOG_PATH   = os.path.join(WORK_DIR, "bho_hin_mt", "loss_log.json")
    STEP_LOSS_PATH  = os.path.join(WORK_DIR, "bho_hin_mt", "train_loss_steps.csv")

    os.makedirs(MODEL_BASE_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Write CSV header only when starting fresh
    if not args.resume:
        with open(STEP_LOSS_PATH, "w") as f:
            f.write("step,loss\n")

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        step_loss_buffer = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=True)
        for batch in pbar:
            try:
                input_ids      = batch["input_ids"].to(get_device(model))
                attention_mask = batch["attention_mask"].to(get_device(model))
                labels         = batch["labels"].to(get_device(model))

                with autocast("cuda", enabled=CUDA):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss.mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                all_train_losses.append(loss_val)
                global_step += 1

                pbar.set_postfix({"loss": f"{np.mean(epoch_losses[-100:]):.4f}"})
                step_loss_buffer.append((global_step, loss_val))

                if len(step_loss_buffer) >= 100:
                    flush_step_losses(STEP_LOSS_PATH, step_loss_buffer)

                if global_step % args.checkpoint_every == 0:
                    flush_step_losses(STEP_LOSS_PATH, step_loss_buffer)
                    ckpt_path = os.path.join(CHECKPOINT_PATH, f"step_{global_step:07d}")
                    save_model(
                        ckpt_path, model, tokenizer,
                        global_step, epoch, optimizer, scheduler, scaler, loss_log, best_bleu, best_chrf,
                    )

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                print(f"  [step {global_step}] RuntimeError: {e}")
                continue

        flush_step_losses(STEP_LOSS_PATH, step_loss_buffer)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                input_ids      = batch["input_ids"].to(get_device(model))
                attention_mask = batch["attention_mask"].to(get_device(model))
                labels         = batch["labels"].to(get_device(model))

                with autocast("cuda", enabled=CUDA):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                val_losses.append(outputs.loss.mean().item())

        train_loss_epoch = np.mean(epoch_losses)
        val_loss_epoch   = np.mean(val_losses)
        bleu, chrf = evaluate_generation_metrics(
            model=unwrap(model),
            tokenizer=tokenizer,
            dev_loader=dev_loader,
            tgt_lang=args.tgt_lang,
            device=get_device(model),
            max_new_tokens=args.max_length,
            max_batches=args.eval_max_batches,
        )
        print(f"\nEpoch {epoch:02d} | train_loss={train_loss_epoch:.4f}  val_loss={val_loss_epoch:.4f}")
        print(f"Epoch {epoch:02d} | BLEU={bleu:.2f}  chrF={chrf:.2f}")

        loss_log.append({
            "epoch":      epoch,
            "train_loss": round(float(train_loss_epoch), 6),
            "val_loss":   round(float(val_loss_epoch),   6),
            "bleu":       round(float(bleu), 4),
            "chrf":       round(float(chrf), 4),
            "steps":      global_step,
        })
        with open(LOSS_LOG_PATH, "w") as f:
            json.dump(loss_log, f, indent=2)
        print(f"Loss log saved → {LOSS_LOG_PATH}")

        epoch_save_path = f"{MODEL_BASE_PATH}_epoch{epoch}"
        save_model(
            epoch_save_path, model, tokenizer,
            global_step, epoch, optimizer, scheduler, scaler, loss_log, best_bleu, best_chrf,
        )

        if bleu > best_bleu:
            best_bleu = bleu
            best_chrf = chrf
            best_path = f"{MODEL_BASE_PATH}_best"
            save_model(
                best_path, model, tokenizer,
                global_step, epoch, optimizer, scheduler, scaler, loss_log, best_bleu, best_chrf,
            )
            print(f"Best checkpoint updated → {best_path}")

        cleanup()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
