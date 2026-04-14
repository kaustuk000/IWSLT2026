"""
train.py
--------
Training script for Inter-connection Bhojpuri → Hindi ST.

Launch via the cluster runner or directly:

    python train.py \
        --asr_url       "Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60" \
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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data_loading import DEFAULT_DATA_DIR, get_dataloaders
from model_loading import build_model, build_processor_and_tokenizer
from pipeline import SpeechTranslationPipeline


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Model sources
    p.add_argument("--asr_url", "--wav2vec2_url", dest="asr_url", required=True,
                   help="Hugging Face repo/URL, FileSender URL, or local path/tarball for the ASR encoder")
    p.add_argument("--nllb_url", required=True,
                   help="Direct FileSender file URL or local path/tarball for NLLB")
    p.add_argument("--cache_dir", default="./model_cache",
                   help="Local directory to cache downloaded/extracted models")

    # Data
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help="Dataset root or clone target. If missing, the repo at "
                        "https://github.com/shashwatup9k/iwslt2026_bho-hi is downloaded.")
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_split",  default="dev")

    # Architecture
    p.add_argument("--aggregation_layers", nargs="+", type=int,
                   default=[16, 20, 24, 31],
                   help="ASR encoder layer indices to aggregate. "
                        "Pass 0 alone to use ALL layers.")
    p.add_argument("--adapter_type", choices=["length", "m_adapter"], default="length",
                   help="Bridge from ASR encoder to NLLB decoder")
    p.add_argument("--adapter_stride",    type=int, default=2)
    p.add_argument("--adapter_num_convs", type=int, default=2)
    p.add_argument("--m_adapter_layers",  type=int, default=2)
    p.add_argument("--m_adapter_heads",   type=int, default=8)
    p.add_argument("--m_adapter_dropout", type=float, default=0.1)
    p.add_argument("--use_aux_ctc",       action="store_true",
                   help="Enable auxiliary target-side CTC loss on speech representations")
    p.add_argument("--aux_ctc_weight",    type=float, default=0.3,
                   help="Weight for the auxiliary target-side CTC loss when enabled")

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
    p.add_argument("--val_loss_every",  type=int,   default=5,
                   help="Run validation loss every N epochs (0=disable)")
    p.add_argument("--eval_beams",      type=int,   default=4)
    p.add_argument("--eval_repetition_penalty", type=float, default=1.0,
                   help="Generation-time repetition penalty for validation decoding")
    p.add_argument("--eval_no_repeat_ngram_size", type=int, default=0,
                   help="Generation-time no-repeat n-gram size for validation decoding (0=disabled)")
    p.add_argument("--eval_max_batches",type=int,   default=None)
    p.add_argument("--unfreeze_encoder_top", type=int, default=0,
                   help="Unfreeze top N encoder layers at epoch epochs//2 (0=never)")
    p.add_argument("--unfreeze_decoder_top", type=int, default=0,
                   help="Use targeted decoder adaptation: unfreeze encoder-attn in all decoder layers, "
                        "all decoder layer norms, and the top N full decoder layers (0=keep decoder frozen)")
    p.add_argument("--unfreeze_decoder_lora", action="store_true",
                   help="Attach LoRA adapters to the NLLB decoder and train only those adapter weights")
    p.add_argument("--train_decoder", action="store_false", dest="freeze_decoder",
                   help="Train the full decoder instead of freezing it by default")
    p.set_defaults(freeze_decoder=True)

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


@torch.no_grad()
def compute_validation_loss(
    model,
    dataloader,
    device,
    *,
    use_amp: bool,
    amp_dtype,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        encoder_inputs = batch["encoder_inputs"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out = model(
                encoder_inputs=encoder_inputs,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += out.loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


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
        args.asr_url, args.nllb_url, args.cache_dir
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
        asr_url=args.asr_url,
        nllb_url=args.nllb_url,
        aggregation_layers=agg_layers,
        cache_dir=args.cache_dir,
        adapter_type=args.adapter_type,
        adapter_stride=args.adapter_stride,
        adapter_num_convs=args.adapter_num_convs,
        m_adapter_layers=args.m_adapter_layers,
        m_adapter_heads=args.m_adapter_heads,
        m_adapter_dropout=args.m_adapter_dropout,
        aux_ctc_weight=args.aux_ctc_weight if args.use_aux_ctc else 0.0,
        aux_special_token_ids=tokenizer.all_special_ids,
        freeze_encoder=True,
        freeze_decoder=args.freeze_decoder,
        decoder_lora=args.unfreeze_decoder_lora,
    ).to(device)
    if args.unfreeze_decoder_top > 0:
        model.unfreeze_decoder_adaptation_params(args.unfreeze_decoder_top)
    if args.unfreeze_decoder_lora:
        model.set_decoder_lora_trainable(True)
    if args.unfreeze_decoder_top > 0 or args.unfreeze_decoder_lora:
        model.print_param_summary()

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

    # ---- log header ----
    with open(log_path, "w") as f:
        f.write("epoch,global_step,train_loss,val_loss,lr_main,lr_agg,val_bleu,val_chrfpp\n")

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
            encoder_inputs = batch["encoder_inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out  = model(encoder_inputs=encoder_inputs,
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
        run_val_loss = (
            eval_loader is not None
            and args.val_loss_every > 0
            and epoch % args.val_loss_every == 0
        )
        run_bleu = (
            eval_loader is not None
            and (
                (args.eval_every > 0 and epoch % args.eval_every == 0)
                or run_val_loss
            )
        )

        val_loss = -1.0
        if run_val_loss:
            val_loss = compute_validation_loss(
                model,
                eval_loader,
                device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_batches=args.eval_max_batches,
            )
            print(f"[eval] Validation loss = {val_loss:.4f}")

        bleu = -1.0
        chrfpp = -1.0
        if run_bleu:
            pipe = SpeechTranslationPipeline(
                model=model, processor=processor, tokenizer=tokenizer,
                device=device,
                num_beams=args.eval_beams,
                repetition_penalty=args.eval_repetition_penalty,
                no_repeat_ngram_size=args.eval_no_repeat_ngram_size,
            )
            metrics = pipe.evaluate(eval_loader, max_batches=args.eval_max_batches)
            bleu = metrics["bleu"]
            chrfpp = metrics["chrfpp"]
            pipe.print_layer_weights()

        # ---- logging ----
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{global_step},{avg_loss:.5f},{val_loss:.5f},"
                f"{lr_m:.2e},{lr_a:.2e},{bleu:.2f},{chrfpp:.2f}\n"
            )

        # ---- checkpointing ----
        torch.save(model.state_dict(), output_dir / "latest_model.pt")
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"[train] Best BLEU={best_bleu:.2f} saved.")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch:03d}.pt")

    print(f"\n[train] Done. Best BLEU: {best_bleu:.2f}")
    print(f"[train] Checkpoints: {output_dir}")


if __name__ == "__main__":
    train(parse_args())
