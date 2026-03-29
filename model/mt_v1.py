import os
import json
import warnings
import torch
import sys, subprocess
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
import multiprocess.resource_tracker as _rt

# ── Silence noisy warnings ────────────────────────────────────────────────

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
_rt.ResourceTracker.__del__ = lambda self: None
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers.data.data_collator",
)
torch.backends.cudnn.benchmark = True

# ── Working directory ─────────────────────────────────────────────────────

WORK_DIR = os.getcwd()
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f"Working dir : {WORK_DIR}")

# ── Fixed Parameters ──────────────────────────────────────────────────────

MT_train = os.path.join(WORK_DIR, "data", "MT_data", "train.csv")
MT_dev   = os.path.join(WORK_DIR, "data", "MT_data", "dev.csv")

BATCH_SIZE       = 16
MAX_LENGTH       = 128
EPOCHS           = 1
WARMUP_STEPS     = 500
CHECKPOINT_EVERY = 1000

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

def save_model(path):
    os.makedirs(path, exist_ok=True)
    unwrap(model).save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved → {path}")

# ── GPU info ──────────────────────────────────────────────────────────────

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

# ── Tokenizer ─────────────────────────────────────────────────────────────

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = NllbTokenizer.from_pretrained(
    model_name,
    src_lang="bho_Deva",
    tgt_lang="hin_Deva",
)

# ── Tokenize function ─────────────────────────────────────────────────────

def tokenize(batch, tokenizer, max_length=128):
    tokenizer.src_lang = "bho_Deva"
    enc = tokenizer(
        batch["bho"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )

    tokenizer.src_lang = "hin_Deva"
    dec = tokenizer(
        batch["hin"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    tokenizer.src_lang = "bho_Deva"

    labels = [
        [-100 if t == tokenizer.pad_token_id else t for t in ids]
        for ids in dec["input_ids"]
    ]

    return {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels":         labels,
    }

# ── Dataset + DataLoader ──────────────────────────────────────────────────

train_ds = load_dataset("csv", data_files=MT_train, split="train")
dev_ds   = load_dataset("csv", data_files=MT_dev,   split="train")

_tokenize = partial(tokenize, tokenizer=tokenizer, max_length=MAX_LENGTH)

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
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=CUDA,
    drop_last=True,
    collate_fn=collator,
)
dev_loader = DataLoader(
    dev_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=CUDA,
    collate_fn=collator,
)

print(f"Train samples : {len(train_ds):,}")
print(f"Dev   samples : {len(dev_ds):,}")
print(f"Train batches : {len(train_loader):,}")
print(f"Dev   batches : {len(dev_loader):,}")

# ── Model ─────────────────────────────────────────────────────────────────

cleanup()

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(DEVICE)
print(f"Model device  : {next(model.parameters()).device}")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    model = torch.nn.DataParallel(model)

# ── Optimizer & Scheduler ─────────────────────────────────────────────────

optimizer = Adafactor(
    [p for p in unwrap(model).parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)

total_steps = EPOCHS * len(train_loader)
scheduler   = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS)
scaler      = GradScaler("cuda", enabled=CUDA)

print(f"Total training steps : {total_steps:,}")

# ── Save paths ────────────────────────────────────────────────────────────

BASE_MODEL_NAME = "model-mt-v1"
MODEL_BASE_PATH = os.path.join(WORK_DIR, "bho_hin_mt", BASE_MODEL_NAME)
CHECKPOINT_PATH = os.path.join(WORK_DIR, "bho_hin_mt", "checkpoints")
LOSS_LOG_PATH   = os.path.join(WORK_DIR, "bho_hin_mt", "loss_log.json")
STEP_LOSS_PATH  = os.path.join(WORK_DIR, "bho_hin_mt", "train_loss_steps.csv")

os.makedirs(MODEL_BASE_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# write CSV header
with open(STEP_LOSS_PATH, "w") as f:
    f.write("step,loss\n")

# ── Training Loop ─────────────────────────────────────────────────────────

all_train_losses = []
loss_log         = []
global_step      = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=True)
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

            # write step loss to CSV immediately
            with open(STEP_LOSS_PATH, "a") as f:
                f.write(f"{global_step},{loss_val:.6f}\n")

            if global_step % CHECKPOINT_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_PATH, f"step_{global_step:07d}")
                save_model(ckpt_path)

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            print(f"  [step {global_step}] RuntimeError: {e}")
            continue

    # ── Validation ────────────────────────────────────────────
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=False):
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
    print(f"\nEpoch {epoch:02d} | train_loss={train_loss_epoch:.4f}  val_loss={val_loss_epoch:.4f}")

    # save epoch summary to JSON
    loss_log.append({
        "epoch":      epoch,
        "train_loss": round(float(train_loss_epoch), 6),
        "val_loss":   round(float(val_loss_epoch), 6),
        "steps":      global_step,
    })
    with open(LOSS_LOG_PATH, "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Loss log saved → {LOSS_LOG_PATH}")

    epoch_save_path = f"{MODEL_BASE_PATH}_epoch{epoch}"
    save_model(epoch_save_path)

    cleanup()

print("\nTraining complete.")