import os
import argparse
import subprocess
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sacrebleu.metrics import BLEU

from data  import BhoHinDataset, make_collate_fn
from model import MMSEncoder, QFormer


TRANSLATE_PROMPT = "<|user|>\nनीचे दी गई सामग्री का हिंदी में अनुवाद करें।\n<|assistant|>\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="iwslt2026_bho-hi/iwslt2024-2025_bho-hi")
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--hf_token",      required=True)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--grad_accum",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--max_label_len", type=int,   default=128)
    p.add_argument("--num_queries",   type=int,   default=80)
    p.add_argument("--log_every",     type=int,   default=100)
    p.add_argument("--bleu_batches",  type=int,   default=50)
    p.add_argument("--resume",        default=None)
    return p.parse_args()


def clone_data(data_dir):
    repo = "https://github.com/shashwatup9k/iwslt2026_bho-hi.git"
    root = data_dir.split("/")[0]
    if not os.path.exists(root):
        print(f"Cloning dataset from {repo}...")
        subprocess.run(["git", "clone", repo], check=True)
    else:
        print("Dataset already exists, skipping clone.")


def get_prompt_embeds_cached(tokenizer, llm, device):
    ids = tokenizer(TRANSLATE_PROMPT, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    embed_fn = llm.base_model.model.model.embed_tokens
    with torch.no_grad():
        embeds = embed_fn(ids).to(torch.bfloat16)
    return embeds


def forward_pass(batch, mms, qformer, llm, prompt_embeds_1, device):
    input_values   = batch["input_values"].to(device, dtype=torch.float16)
    attention_mask = batch["attention_mask"].to(device)
    labels         = batch["labels"].to(device)
    B              = input_values.size(0)

    enc_out       = mms(input_values, attention_mask)
    speech_embeds = qformer(enc_out.to(torch.float32)).to(torch.bfloat16)
    prompt_embeds = prompt_embeds_1.expand(B, -1, -1)

    embed_fn      = llm.base_model.model.model.embed_tokens
    safe_labels   = labels.clone()
    safe_labels[safe_labels == -100] = 0
    label_embeds  = embed_fn(safe_labels).to(torch.bfloat16)

    inputs_embeds = torch.cat([prompt_embeds, speech_embeds, label_embeds], dim=1)

    ignore      = torch.full((B, prompt_embeds.size(1) + speech_embeds.size(1)),
                             -100, dtype=torch.long, device=device)
    full_labels = torch.cat([ignore, labels], dim=1)

    prompt_mask = torch.ones((B, prompt_embeds.size(1)), device=device)
    speech_mask = torch.ones((B, speech_embeds.size(1)), device=device)
    label_mask  = (labels != -100).long()
    full_mask   = torch.cat([prompt_mask, speech_mask, label_mask], dim=1)

    return llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels).loss


def evaluate_bleu(dev_loader, mms, qformer, llm, tokenizer, prompt_embeds_1, device, num_batches=50):
    qformer.eval()
    llm.eval()
    hypotheses, references = [], []

    with torch.no_grad():
        for i, batch in enumerate(dev_loader):
            if num_batches and i >= num_batches:
                break
            input_values   = batch["input_values"].to(device, dtype=torch.float16)
            attention_mask = batch["attention_mask"].to(device)
            B              = input_values.size(0)

            enc_out       = mms(input_values, attention_mask)
            speech_embeds = qformer(enc_out.to(torch.float32)).to(torch.bfloat16)
            prompt_embeds = prompt_embeds_1.expand(B, -1, -1)
            inputs_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)

            out = llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)
            safe  = batch["labels"].clone().masked_fill(batch["labels"] == -100, tokenizer.pad_token_id)
            refs  = tokenizer.batch_decode(safe, skip_special_tokens=True)
            hypotheses.extend(preds)
            references.extend(refs)

    score = BLEU().corpus_score(hypotheses, [references])
    print(f"  BLEU ({len(hypotheses)} samples): {score}")
    return score.score


def main():
    args = parse_args()
    clone_data(args.data_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    print("Loading MMS encoder...")
    mms = MMSEncoder().to(device)

    print("Loading Airavata in 8-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        "ai4bharat/Airavata",
        token=args.hf_token,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # required before adding LoRA to a quantized model
    llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
    llm = get_peft_model(llm, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none",
    ))
    llm.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/Airavata", token=args.hf_token, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets...")
    train_ds = BhoHinDataset(args.data_dir, "train")
    dev_ds   = BhoHinDataset(args.data_dir, "dev")

    collate_fn   = make_collate_fn(mms.processor, tokenizer, args.max_label_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    qformer = QFormer(num_queries=args.num_queries).to(device)

    start_epoch = 0
    best_bleu   = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        qformer.load_state_dict(ckpt["qformer"])
        llm.load_state_dict(ckpt["lora"], strict=False)
        start_epoch = ckpt["epoch"] + 1
        best_bleu   = ckpt.get("bleu", 0.0)
        print(f"Resumed from epoch {start_epoch}, best BLEU {best_bleu:.2f}")

    trainable = list(qformer.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    prompt_embeds_1 = get_prompt_embeds_cached(tokenizer, llm, device)

    for epoch in range(start_epoch, args.epochs):
        qformer.train()
        llm.train()
        optimizer.zero_grad()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            loss = forward_pass(batch, mms, qformer, llm, prompt_embeds_1, device) / args.grad_accum
            loss.backward()
            total_train_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                avg = total_train_loss / (step + 1)
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss {loss.item()*args.grad_accum:.4f} | Avg {avg:.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        qformer.eval()
        llm.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                total_val_loss += forward_pass(batch, mms, qformer, llm, prompt_embeds_1, device).item()

        avg_val_loss = total_val_loss / len(dev_loader)
        scheduler.step()

        print(f"\nEpoch {epoch+1} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        bleu = evaluate_bleu(dev_loader, mms, qformer, llm, tokenizer,
                             prompt_embeds_1, device, args.bleu_batches)

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save({
                "epoch":     epoch,
                "qformer":   qformer.state_dict(),
                "lora":      {k: v for k, v in llm.state_dict().items() if "lora" in k},
                "optimizer": optimizer.state_dict(),
                "val_loss":  avg_val_loss,
                "bleu":      best_bleu,
            }, f"{args.save_dir}/best_checkpoint.pt")
            print(f"  → Saved best (BLEU={best_bleu:.2f})\n")
        else:
            print()

    print("Training complete.")


if __name__ == "__main__":
    main()