"""
train.py
--------
Training script for Inter-connection Bhojpuri → Hindi ST.

Launch via setup_and_run.sh or directly:

    python train.py \
        --wav2vec2_url  "Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60" \
        --nllb_url      "https://filesender.cesnet.cz/download.php?token=BBB&files_ids=12345" \
        --data_dir      /path/to/iwslt2026_bho_hi \
        --aggregation_layers 6 12 18 24 \
        --output_dir    ./checkpoints
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data_loading import get_dataloaders
from model_loading import build_model, build_processor_and_tokenizer
from pipeline import SpeechTranslationPipeline


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Model sources
    p.add_argument("--wav2vec2_url", required=True,
                   help="Hugging Face repo/URL, FileSender URL, or local path/tarball for wav2vec2")
    p.add_argument("--nllb_url", required=True,
                   help="Direct FileSender file URL or local path/tarball for NLLB")
    p.add_argument("--cache_dir", default="./model_cache",
                   help="Local directory to cache downloaded/extracted models")

    # Data
    p.add_argument("--data_dir", default= "./IWSLT2026/iwslt2026_bho_hi",
                   help="Dataset root or clone target. If missing, the repo at "
                        "https://github.com/shashwatup9k/iwslt2026_bho-hi is downloaded.")
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_split",  default="dev")

    # Architecture
    p.add_argument("--aggregation_layers", nargs="+", type=int,
                   default=[6, 8, 10, 12],
                   help="Wav2vec2 layer indices to aggregate. "
                        "Pass 0 alone to use ALL layers.")
    p.add_argument("--adapter_stride",    type=int, default=2)
    p.add_argument("--adapter_num_convs", type=int, default=2)

    # Training
    p.add_argument("--output_dir",      default="./checkpoints")
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--grad_accum",      type=int,   default=8)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--aggregator_lr",   type=float, default=1e-3)
    p.add_argument("--warmup_ratio",    type=float, default=0.05)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dtype",           choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--eval_every",      type=int,   default=1)
    p.add_argument("--eval_beams",      type=int,   default=4)
    p.add_argument("--eval_max_batches",type=int,   default=None)
    p.add_argument("--unfreeze_encoder_top", type=int, default=0,
                   help="Unfreeze top N encoder layers at epoch epochs//2 (0=never)")
    p.add_argument("--freeze_decoder", action="store_true",
                   help="Freeze NLLB decoder weights to reduce GPU memory usage")

    return p.parse_args()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.csv"

    # ---- precision ----
    use_amp   = args.dtype in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    scaler    = GradScaler() if (use_amp and args.dtype == "fp16") else None

    # ---- processor / tokenizer / data ----
    print("[train] Loading processor and tokenizer...")
    processor, tokenizer = build_processor_and_tokenizer(
        args.wav2vec2_url, args.nllb_url, args.cache_dir
    )

    print("[train] Building dataloaders...")
    train_loader, eval_loader = get_dataloaders(
        base_path=args.data_dir,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )

    # ---- model ----
    agg_layers = [] if args.aggregation_layers == [0] else args.aggregation_layers
    print(f"[train] Aggregation layers: {agg_layers if agg_layers else 'ALL'}")

    model = build_model(
        wav2vec2_url=args.wav2vec2_url,
        nllb_url=args.nllb_url,
        aggregation_layers=agg_layers,
        cache_dir=args.cache_dir,
        adapter_stride=args.adapter_stride,
        adapter_num_convs=args.adapter_num_convs,
        freeze_encoder=True,
        freeze_decoder=args.freeze_decoder,
    ).to(device)

    # ---- optimizer: separate LR for aggregator ----
    agg_ids    = set(id(p) for p in model.aggregator.parameters())
    agg_params = list(model.aggregator.parameters())
    rest_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in agg_ids]

    optimizer = AdamW(
        [
            {"params": agg_params,  "lr": args.aggregator_lr, "name": "aggregator"},
            {"params": rest_params, "lr": args.lr,             "name": "main"},
        ],
        weight_decay=args.weight_decay,
    )

    # ---- scheduler ----
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(total_steps * args.warmup_ratio)
    scheduler       = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[train] Steps: {total_steps} total, {warmup_steps} warmup")

    # ---- loss ----
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=args.label_smoothing)

    # ---- log header ----
    with open(log_path, "w") as f:
        f.write("epoch,global_step,avg_loss,lr_main,lr_agg,bleu\n")

    best_bleu    = -1.0
    global_step  = 0

    # ---- epoch loop ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_values   = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out  = model(input_values=input_values,
                             attention_mask=attention_mask,
                             labels=labels)
                loss = out.loss / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            n_batches  += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    lr_m = optimizer.param_groups[1]["lr"]
                    lr_a = optimizer.param_groups[0]["lr"]
                    print(f"  step {global_step:5d} | loss {epoch_loss/n_batches:.4f} "
                          f"| lr_main {lr_m:.2e} | lr_agg {lr_a:.2e}")

        # ---- mid-training encoder unfreeze ----
        if args.unfreeze_encoder_top > 0 and epoch == args.epochs // 2:
            model.unfreeze_encoder_top_layers(args.unfreeze_encoder_top)
            new_params = [p for p in model.encoder.parameters() if p.requires_grad]
            optimizer.add_param_group(
                {"params": new_params, "lr": args.lr * 0.1, "name": "encoder_top"}
            )

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_m     = optimizer.param_groups[1]["lr"]
        lr_a     = optimizer.param_groups[0]["lr"]
        elapsed  = time.time() - t0
        print(f"[Epoch {epoch:3d}/{args.epochs}] loss={avg_loss:.4f} | "
              f"lr_main={lr_m:.2e} | lr_agg={lr_a:.2e} | {elapsed:.1f}s")

        # ---- eval ----
        bleu = -1.0
        if eval_loader and epoch % args.eval_every == 0:
            pipe = SpeechTranslationPipeline(
                model=model, processor=processor, tokenizer=tokenizer,
                device=device, num_beams=args.eval_beams,
            )
            bleu = pipe.evaluate(eval_loader, max_batches=args.eval_max_batches)
            pipe.print_layer_weights()

        # ---- logging ----
        with open(log_path, "a") as f:
            f.write(f"{epoch},{global_step},{avg_loss:.5f},{lr_m:.2e},{lr_a:.2e},{bleu:.2f}\n")

        # ---- checkpointing ----
        torch.save(model.state_dict(), output_dir / "latest_model.pt")
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"[train] ✓ Best BLEU={best_bleu:.2f} saved.")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch:03d}.pt")

    print(f"\n[train] Done. Best BLEU: {best_bleu:.2f}")
    print(f"[train] Checkpoints: {output_dir}")


if __name__ == "__main__":
    train(parse_args())
