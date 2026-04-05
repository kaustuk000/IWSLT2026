import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # ← helps catch exact assert error location

import torch

if torch.cuda.device_count() > 1:
    torch.nn.DataParallel = lambda model, **kwargs: model
    print(f"DataParallel disabled — was seeing {torch.cuda.device_count()} GPUs")

print(f"Visible GPUs : {torch.cuda.device_count()}")
print(f"Active GPU   : {torch.cuda.get_device_name(0)}")
print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("done")




import gc, math, json, itertools, subprocess, tempfile
import re
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from scipy.stats import t as t_dist
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
import librosa
from jiwer import wer, cer, process_words
from IPython.display import Audio, display
import warnings
warnings.filterwarnings("ignore")

print("Imports ✓")
print(f"Torch  : {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")



MODEL_NAME    = "openai/whisper-large-v3"
DATASET_NAME  = "ai4bharat/Rural_Women_Bhojpuri"
SAMPLE_RATE   = 16000
MAX_AUDIO_SEC = 20          
SEEDS         = [42, 1337, 2024]
WORK_DIR      = os.environ.get("WORK_DIR", os.getcwd())
OUTPUT_DIR    = os.path.join(WORK_DIR, "whisper-bhojpuri")
SAVE_PATH     = os.path.join(WORK_DIR, "whisper-bhojpuri-final")
LM_ARPA_PATH  = os.path.join(WORK_DIR, "bhojpuri_lm.arpa")

SPLIT_REAL      = "train_real"
SPLIT_SYNTHETIC = "train_synthetic"
SPLIT_BENCHMARK = "benchmark"

REAL_SAMPLES      = 400
SYNTHETIC_SAMPLES = 200
TEST_SIZE         = 20
VAL_SIZE          = 20
TRAIN_SIZE        = REAL_SAMPLES + SYNTHETIC_SAMPLES

print(f"Train : {TRAIN_SIZE}")
print(f"Val   : {VAL_SIZE}")
print(f"Test  : {TEST_SIZE}  ← only touched ONCE at the very end")



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEEDS[0])
print(f"Seed set to {SEEDS[0]} ✓")


print("Loading train_real…")
ds_real = load_dataset(DATASET_NAME, split=SPLIT_REAL, streaming=True)
real_samples = list(ds_real.take(REAL_SAMPLES))  
print(f"  train_real      : {len(real_samples)} samples")

print("Loading train_synthetic…")
ds_syn = load_dataset(DATASET_NAME, split=SPLIT_SYNTHETIC, streaming=True)
syn_samples = list(ds_syn.take(SYNTHETIC_SAMPLES))
print(f"  train_synthetic : {len(syn_samples)} samples")

print("Loading benchmark…")
ds_bench = load_dataset(DATASET_NAME, split=SPLIT_BENCHMARK, streaming=True)
bench_samples = list(ds_bench.take(VAL_SIZE + TEST_SIZE))  
print(f"  benchmark       : {len(bench_samples)} samples")

random.seed(42)
random.shuffle(real_samples)
random.shuffle(syn_samples)
random.shuffle(bench_samples)

train_samples = real_samples + syn_samples

mid = len(bench_samples) // 2
val_samples  = bench_samples[:mid]
test_samples = bench_samples[mid:]

REAL_SAMPLES      = len(real_samples)
SYNTHETIC_SAMPLES = len(syn_samples)
TRAIN_SIZE        = len(train_samples)
VAL_SIZE          = len(val_samples)
TEST_SIZE         = len(test_samples)

print(f"\nFinal split sizes:")
print(f"  Train : {TRAIN_SIZE}  ({REAL_SAMPLES} real + {SYNTHETIC_SAMPLES} synthetic)")
print(f"  Val   : {VAL_SIZE}")
print(f"  Test  : {TEST_SIZE}  ← do NOT touch until final eval")



DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
}

def normalize_text(text: str) -> str:
    text = text.strip().lower()
    for dv, en in DEVANAGARI_DIGITS.items():
        text = text.replace(dv, en)
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

print("Normalization check:", normalize_text("हेलो, World! ३४५"))



def load_audio(sample: dict) -> np.ndarray:
    array = np.array(sample["audio"]["array"], dtype=np.float32)
    src_sr = sample["audio"]["sampling_rate"]
    if src_sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=src_sr, target_sr=SAMPLE_RATE)
    return array[:SAMPLE_RATE * MAX_AUDIO_SEC]

print("Audio loader ✓")

def play_audio(sample):
    array = load_audio(sample)
    display(Audio(array, rate=SAMPLE_RATE))

play_audio(train_samples[0])



def add_gaussian_noise(array: np.ndarray, snr_db_range=(15, 30)) -> np.ndarray:
    snr_db    = random.uniform(*snr_db_range)
    sig_power = np.mean(array ** 2) + 1e-9
    noise_std = math.sqrt(sig_power / (10 ** (snr_db / 10)))
    noise     = np.random.randn(len(array)).astype(np.float32) * noise_std
    return array + noise

def speed_perturb(array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    factor = random.choice([0.9, 1.0, 1.0, 1.1])
    if factor != 1.0:
        array = librosa.effects.time_stretch(array, rate=factor)
    return array[:sr * MAX_AUDIO_SEC]

def augment_waveform(array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if random.random() < 0.5:
        array = speed_perturb(array, sr)
    if random.random() < 0.5:
        array = add_gaussian_noise(array)
    return array

def spec_augment(
    features: torch.Tensor,
    num_freq_masks: int = 2,
    freq_mask_param: int = 13,
    num_time_masks: int = 2,
    time_mask_param: int = 20,
    replace_with_zero: bool = False,
) -> torch.Tensor:
    cloned  = features.clone()
    squeezed = (cloned.dim() == 3)
    if squeezed:
        cloned = cloned.squeeze(0)

    n_freq, n_time = cloned.shape
    fill = 0.0 if replace_with_zero else cloned.mean().item()

    for _ in range(num_freq_masks):
        f  = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(n_freq - f, 0))
        cloned[f0 : f0 + f, :] = fill

    speech_end = int((cloned != cloned.min()).any(dim=0).float().argmin().item())
    speech_end = speech_end if speech_end > 0 else n_time

    for _ in range(num_time_masks):
        t  = random.randint(0, min(time_mask_param, speech_end))
        t0 = random.randint(0, max(speech_end - t, 0))
        cloned[:, t0 : t0 + t] = fill

    if squeezed:
        cloned = cloned.unsqueeze(0)
    return cloned

print("Augmentation defined ✓")


print("Loading processor…")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.set_prefix_tokens(language="hi", task="transcribe")

print("Loading model…")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.use_cache = False

forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
model.config.forced_decoder_ids            = forced_decoder_ids
model.generation_config.forced_decoder_ids = forced_decoder_ids
print("Model loaded ✓")


print("\nComputing zero-shot baseline on val set…")
model.eval()
model = model.to(torch.float32)

def transcribe_single(sample: dict) -> Tuple[str, str]:
    array = load_audio(sample)
    inputs = processor(
        array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True
    )
    input_features = inputs.input_features.to(model.device).to(model.dtype)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        pred_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            task="transcribe",
            language="hi",
            num_beams=2,   
        )

    pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    ref  = sample["text"]
    return normalize_text(pred), normalize_text(ref)


BASELINE_SAMPLES = min(100, len(val_samples))   
baseline_preds, baseline_refs = [], []

for s in tqdm(val_samples[:BASELINE_SAMPLES], desc="Baseline eval"):
    pred, ref = transcribe_single(s)
    baseline_preds.append(pred)
    baseline_refs.append(ref)

baseline_wer = wer(baseline_refs, baseline_preds)
baseline_cer = cer(baseline_refs, baseline_preds)

print("\n" + "="*55)
print("  ZERO-SHOT BASELINE (Whisper-large-v3, no FT)")
print("="*55)
print(f"  WER : {baseline_wer*100:.2f}%")
print(f"  CER : {baseline_cer*100:.2f}%")
print("="*55)




lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model = model.to(torch.float32)
print("Model cast to float32 ✓")

# ──re-apply after LoRA wrapping ──────────────────────
model.config.forced_decoder_ids            = forced_decoder_ids
model.generation_config.forced_decoder_ids = forced_decoder_ids
print("forced_decoder_ids re-applied after LoRA ✓")

model.enable_input_require_grads()
model.gradient_checkpointing_enable()
print("Gradient checkpointing ✓")





VOCAB_SIZE = processor.tokenizer.vocab_size   # ← used to clamp labels

class BhojpuriDataset(Dataset):
    def __init__(self, samples: list, augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        array  = load_audio(sample)
        if self.augment:
            array = augment_waveform(array)
        inputs = processor(
            array,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True
        )
        features = inputs.input_features[0]
        if self.augment:
            features = spec_augment(features)

        label_ids = processor.tokenizer(
            normalize_text(sample["text"]),
            return_tensors="pt"
        ).input_ids[0].tolist()

        # ── clamp label IDs to valid vocab range ──────
        label_ids = [
            t for t in label_ids
            if 0 <= t < VOCAB_SIZE
        ]

        return {
            "input_features": features,
            "attention_mask": inputs.attention_mask[0],
            "labels":         label_ids
        }

train_dataset = BhojpuriDataset(train_samples, augment=True)
val_dataset   = BhojpuriDataset(val_samples,   augment=False)
print(f"Train : {len(train_dataset)} ✓")
print(f"Val   : {len(val_dataset)}  ✓")

# Quick sanity check
sample_out = train_dataset[0]
print("Feature shape :", sample_out["input_features"].shape)
print("Label IDs (first 10):", sample_out["labels"][:10])
print("Decoded:", processor.tokenizer.decode(sample_out["labels"]))




@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── input features ──────────────────────────────────
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(torch.float32)

        # ── labels ──────────────────────────────────────────
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt", return_attention_mask=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        # ── clamp to valid vocab range (second safety layer) ──
        valid_mask = labels != -100
        labels[valid_mask] = labels[valid_mask].clamp(0, VOCAB_SIZE - 1)

        batch["labels"] = labels

        decoder_start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        shifted = labels.masked_fill(labels == -100, self.processor.tokenizer.pad_token_id)
        bos_col = torch.full((shifted.size(0), 1), decoder_start, dtype=torch.long)
        batch["decoder_input_ids"] = torch.cat([bos_col, shifted[:, :-1]], dim=1)

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("Data collator ✓")




def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs  = [normalize_text(p) for p in
                  processor.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_strs = [normalize_text(l) for l in
                  processor.batch_decode(label_ids, skip_special_tokens=True)]

    return {
        "wer": round(wer(label_strs, pred_strs), 4),
        "cer": round(cer(label_strs, pred_strs), 4)
    }

print("compute_metrics ✓")




class EpochSummaryCallback(TrainerCallback):
    def __init__(self):
        self.epoch_logs   = []
        self._step_losses = []
        self._step_lr     = []
        self._step_gnorm  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        if "loss"          in logs: self._step_losses.append(logs["loss"])
        if "learning_rate" in logs: self._step_lr.append(logs["learning_rate"])
        if "grad_norm"     in logs: self._step_gnorm.append(logs["grad_norm"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics: return
        epoch     = int(state.epoch)
        avg_loss  = round(float(np.mean(self._step_losses)), 4) if self._step_losses else float("nan")
        last_lr   = round(float(self._step_lr[-1]), 8)          if self._step_lr     else float("nan")
        avg_gnorm = round(float(np.mean(self._step_gnorm)), 4)  if self._step_gnorm  else float("nan")

        row = {
            "Epoch":      epoch,
            "Train Loss": avg_loss,
            "Eval Loss":  round(metrics.get("eval_loss",  float("nan")), 4),
            "WER (%)":    round(metrics.get("eval_wer",   float("nan")) * 100, 2),
            "CER (%)":    round(metrics.get("eval_cer",   float("nan")) * 100, 2),
            "LR":         last_lr,
            "Grad Norm":  avg_gnorm,
        }
        self.epoch_logs.append(row)
        self._step_losses = []; self._step_lr = []; self._step_gnorm = []

        sep = "=" * 65
        print(f"\n{sep}\n  EPOCH {epoch} SUMMARY\n{sep}")
        for k, v in row.items(): print(f"  {k:<14}: {v}")
        print(f"{sep}\n")

    def on_train_end(self, args, state, control, **kwargs):
        if not self.epoch_logs: return
        df   = pd.DataFrame(self.epoch_logs)
        best = df.loc[df["WER (%)"].idxmin()]
        print("\n" + "="*65)
        print("  FULL TRAINING SUMMARY")
        print("="*65)
        print(df.to_string(index=False))
        print(f"\n  ✓ Best Epoch : {int(best['Epoch'])}  "
              f"WER: {best['WER (%)']}%  CER: {best['CER (%)']}%")
        print("="*65)

epoch_callback = EpochSummaryCallback()
print("Callback ✓")




training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,       
    gradient_accumulation_steps=4,      
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,         

    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,

    num_train_epochs=4,               

    fp16=False,                        
    bf16=True,                          
    fp16_full_eval=False,

    predict_with_generate=True,
    generation_max_length=128,         
    generation_num_beams=1,            

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    logging_steps=5,
    logging_first_step=True,
    save_total_limit=1,              
    max_grad_norm=1.0,
    report_to="none",

    label_smoothing_factor=0.1,
)
print("Training args ✓")





os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
gc.collect()

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[epoch_callback],
)

# ── Quick batch sanity check before training ──────────────
print("Sanity checking one batch…")
batch = data_collator([train_dataset[i] for i in range(2)])
for k, v in batch.items():
    print(f"  {k:<22} shape={tuple(v.shape)}  dtype={v.dtype}  "
          f"min={v[v != -100].min().item():.0f}  max={v.max().item():.0f}")
print("Batch OK ✓\n")

print("Starting training…\n")
trainer.train()
print("\nTraining complete ✓")