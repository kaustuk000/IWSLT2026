# ============================================================
# CELL 1 
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"]   = "expandable_segments:True"

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    torch.nn.DataParallel = lambda model, **kwargs: model
    print(f"DataParallel disabled — was seeing {torch.cuda.device_count()} GPUs")

print(f"Visible GPUs : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Active GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Active device: CPU")
 
 
print("done")



# ============================================================
# CELL 2 — Imports
# ============================================================
import os, gc, math, json, itertools, subprocess, tempfile, io
import torch
import re
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from scipy.stats import t as t_dist      
from tqdm.auto import tqdm
from datasets import load_dataset, Audio as HfAudio
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
import torchaudio
import soundfile as sf
from jiwer import wer, cer, process_words
from IPython.display import Audio, display
import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
 
print("Imports ✓")
print(f"Torch  : {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"Device : {DEVICE}")
 
 
# ============================================================
# CELL 3 — Config
# ============================================================
MODEL_NAME    = "openai/whisper-large-v3"
DATASET_NAME  = "ai4bharat/Rural_Women_Bhojpuri"
SAMPLE_RATE   = 16000
MAX_AUDIO_SEC = 30
SEEDS         = [42, 1337, 2024]
WORK_DIR      = os.environ.get("WORK_DIR", os.getcwd())
OUTPUT_DIR    = os.path.join(WORK_DIR, "whisper-bhojpuri")
SAVE_PATH     = os.path.join(WORK_DIR, "whisper-bhojpuri-final")
LM_ARPA_PATH  = os.path.join(WORK_DIR, "bhojpuri_lm.arpa")
RESULTS_TABLE_PATH = os.path.join(WORK_DIR, "results_table.csv")
EPOCH_SUMMARY_PATH = os.path.join(WORK_DIR, "epoch_summary.csv")
 
SPLIT_REAL      = "train_real"
SPLIT_SYNTHETIC = "train_synthetic"
SPLIT_BENCHMARK = "benchmark"

REAL_SAMPLES      = 400   
SYNTHETIC_SAMPLES = 10000   
TEST_SIZE         = 2000    
VAL_SIZE          = 2000   
TRAIN_SIZE        = REAL_SAMPLES + SYNTHETIC_SAMPLES

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16 = torch.cuda.is_available() and not USE_BF16
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16 if USE_FP16 else torch.float32
BASELINE_NUM_BEAMS = 2
EVAL_NUM_BEAMS = 2
DATA_LOADER_WORKERS = min(4, os.cpu_count() or 1)
 
print(f"Train : {TRAIN_SIZE}")
print(f"Val   : {VAL_SIZE}")
print(f"Test  : {TEST_SIZE}  ← only touched ONCE at the very end")
print(f"Work dir    : {WORK_DIR}")
print(f"Model dtype : {MODEL_DTYPE}")
print(f"Workers     : {DATA_LOADER_WORKERS}")
 

# ============================================================
# CELL 4 — Reproducibility
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
set_seed(SEEDS[0])
print(f"Seed set to {SEEDS[0]} ✓")
 
# ============================================================
# CELL 5 — Load + Split Dataset
# ============================================================
def load_streaming_split(split_name: str):
    ds = load_dataset(DATASET_NAME, split=split_name, streaming=True)
    return ds.cast_column("audio", HfAudio(decode=False))

print("Loading train_real…")
ds_real = load_streaming_split(SPLIT_REAL)
real_samples = list(ds_real)   # take all (~400)
print(f"  train_real      : {len(real_samples)} samples")

print("Loading train_synthetic…")
ds_syn = load_streaming_split(SPLIT_SYNTHETIC)
syn_samples = list(ds_syn.take(SYNTHETIC_SAMPLES))
print(f"  train_synthetic : {len(syn_samples)} samples (subset)")

print("Loading benchmark…")
ds_bench = load_streaming_split(SPLIT_BENCHMARK)
bench_samples = list(ds_bench)
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


# ============================================================
# CELL 6 — Text Normalization
# ============================================================
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
 

# ============================================================
# CELL 7 — Audio Loading
# ============================================================
def _decode_audio_blob(audio_blob: dict) -> Tuple[np.ndarray, int]:
    if "array" in audio_blob and audio_blob["array"] is not None:
        return np.asarray(audio_blob["array"], dtype=np.float32), int(audio_blob["sampling_rate"])

    audio_bytes = audio_blob.get("bytes")
    audio_path = audio_blob.get("path")

    if audio_bytes is not None:
        array, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    elif audio_path:
        array, src_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio sample is missing both decoded data and raw storage fields.")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array, int(src_sr)


def load_audio(sample: dict) -> np.ndarray:
    array, src_sr = _decode_audio_blob(sample["audio"])
    if src_sr != SAMPLE_RATE:
        array = torchaudio.functional.resample(
            torch.from_numpy(array),
            orig_freq=src_sr,
            new_freq=SAMPLE_RATE,
        ).numpy()
    return array[:SAMPLE_RATE * MAX_AUDIO_SEC]
 
print("Audio loader ✓")
 
 
def play_audio(sample):
    array = load_audio(sample)
    display(Audio(array, rate=SAMPLE_RATE))
play_audio(train_samples[0])


# ============================================================
# CELL 8 — Data Augmentation

# ============================================================
 
 
def add_gaussian_noise(array: np.ndarray, snr_db_range=(15, 30)) -> np.ndarray:
    """
    Add white Gaussian noise at a uniformly-sampled SNR in [snr_db_range].
    SNR = 10 * log10(signal_power / noise_power).
    """
    snr_db    = random.uniform(*snr_db_range)
    sig_power = np.mean(array ** 2) + 1e-9
    noise_std = math.sqrt(sig_power / (10 ** (snr_db / 10)))
    noise     = np.random.randn(len(array)).astype(np.float32) * noise_std
    return array + noise
 
def speed_perturb(array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Temporal stretch by ±10 %, biased toward no-op (50 % chance)."""
    factor = random.choice([0.9, 1.0, 1.0, 1.1])
    if factor != 1.0:
        array = librosa.effects.time_stretch(array, rate=factor)
    return array[:sr * MAX_AUDIO_SEC]
 
def augment_waveform(array: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply speed perturbation + Gaussian noise (both with 50 % probability each)."""
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

    
    cloned = features.clone()
    squeezed = (cloned.dim() == 3)
    if squeezed:
        cloned = cloned.squeeze(0)      
 
    n_freq, n_time = cloned.shape
    fill = 0.0 if replace_with_zero else cloned.mean().item()
 
    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(n_freq - f, 0))
        cloned[f0 : f0 + f, :] = fill
 
    # Time masking
    speech_end = int((cloned != cloned.min()).any(dim=0).float().argmin().item())
    speech_end = speech_end if speech_end > 0 else n_time  

    for _ in range(num_time_masks):
        t  = random.randint(0, min(time_mask_param, speech_end))
        t0 = random.randint(0, max(speech_end - t, 0))
        cloned[:, t0 : t0 + t] = fill
 
    if squeezed:
        cloned = cloned.unsqueeze(0)
    return cloned
 
print("Augmentation (SpecAugment + noise + speed) defined ✓")
 
 
def test_audio_augmentation(sample, n_trials=3):
    print("TEXT:", sample["text"])
    
    original = load_audio(sample)
    
    print("\n🔊 ORIGINAL:")
    display(Audio(original, rate=SAMPLE_RATE))
    
    for i in range(n_trials):
        aug = augment_waveform(original.copy())
        
        print(f"\n🔊 AUGMENTED #{i+1}:")
        display(Audio(aug, rate=SAMPLE_RATE))

test_audio_augmentation(train_samples[0])


 
# ============================================================
# CELL 9 — Load Model + Processor
# ============================================================
print("Loading processor…")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.set_prefix_tokens(language="hi", task="transcribe")
 
print("Loading model…")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.use_cache = False
 
forced_decoder_ids = processor.get_decoder_prompt_ids(language="hi", task="transcribe")
model.config.forced_decoder_ids            = forced_decoder_ids
model.generation_config.forced_decoder_ids = forced_decoder_ids
model = model.to(device=DEVICE, dtype=MODEL_DTYPE)
print("Model loaded ✓")
 
# ============================================================
# CELL 10 — BASELINE: Zero-shot WER
# ============================================================


print("\nComputing zero-shot baseline on val set…")
model.eval()
model = model.to(device=DEVICE, dtype=MODEL_DTYPE)

def transcribe_single(sample: dict) -> Tuple[str, str]:
    array = load_audio(sample)
    inputs = processor(
        array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True
    )
    input_features = inputs.input_features.to(device=DEVICE, dtype=MODEL_DTYPE)
    attention_mask = inputs.attention_mask.to(DEVICE)

    with torch.inference_mode():
        pred_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            task="transcribe",
            language="hi",
            num_beams=BASELINE_NUM_BEAMS,
        )

    pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    ref  = sample["text"]
    return normalize_text(pred), normalize_text(ref)


BASELINE_SAMPLES = 100 
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

pd.DataFrame([{
    "Model": "Whisper-large-v3 (zero-shot)",
    "WER": round(baseline_wer*100, 2),
    "CER": round(baseline_cer*100, 2),
    "Train Samples": 0
}]).to_csv(RESULTS_TABLE_PATH, index=False)


# ============================================================
# CELL 11 — Apply LoRA
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
 
model = model.to(device=DEVICE, dtype=MODEL_DTYPE)
print(f"Model cast to {MODEL_DTYPE} ✓")

model.enable_input_require_grads()  
model.gradient_checkpointing_enable()
print("Gradient checkpointing ✓")

# ============================================================
# CELL 12 — Preprocessing  (lazy, on-the-fly — no RAM spike)
# ============================================================

class BhojpuriDataset(Dataset):
    """
    Preprocesses each sample on-the-fly during training.
    Keeps only raw samples in RAM (~10x less memory than
    pre-computing all mel spectrograms upfront).
    """
    def __init__(self, samples: list, augment: bool = False):
        self.samples = samples
        self.augment = augment
        self.cache_processed = not augment
        self._cache = {}
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        if self.cache_processed and idx in self._cache:
            cached = self._cache[idx]
            return {
                "input_features": cached["input_features"].clone(),
                "attention_mask": cached["attention_mask"].clone(),
                "labels": cached["labels"][:],
            }
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
        labels = processor.tokenizer(
            normalize_text(sample["text"]),
            return_tensors="pt"
        ).input_ids
        item = {
            "input_features": features,
            "attention_mask": inputs.attention_mask[0],
            "labels":         labels[0].tolist()
        }
        if self.cache_processed:
            self._cache[idx] = {
                "input_features": item["input_features"].clone(),
                "attention_mask": item["attention_mask"].clone(),
                "labels": item["labels"][:],
            }
        return item
train_dataset = BhojpuriDataset(train_samples, augment=True)
val_dataset   = BhojpuriDataset(val_samples,   augment=False)
print(f"Train : {len(train_dataset)} ✓  (lazy — no RAM spike)")
print(f"Val   : {len(val_dataset)}  ✓")
print("Test set will be wrapped at final evaluation.")

# ============================================================
# DEBUG — Inspect one preprocessed sample
# ============================================================

def inspect_sample(dataset, idx=10):
    sample = dataset[idx]

    print("🔹 Feature shape:")
    print(sample["input_features"].shape)  

    print("\n🔹 Label IDs (first 20):")
    print(sample["labels"][:20])

    print("\n🔹 Decoded text:")
    print(processor.tokenizer.decode(sample["labels"]))

    return sample


sample_out = inspect_sample(train_dataset, idx=0)



def show_spectrogram(sample):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))  
    plt.clf()                    

    features = sample["input_features"].numpy()

    plt.imshow(features, aspect='auto', origin='lower')
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.colorbar()

    plt.show()


show_spectrogram(sample_out)

# ============================================================
# CELL 13 — Data Collator  (explicit decoder_input_ids)
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── input features ──────────────────────────────────────
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(MODEL_DTYPE)

        # ── labels ──────────────────────────────────────────────
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt", return_attention_mask=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        # strip BOS if prepended
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        decoder_start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        # Replace -100 padding with pad_token_id for the shift
        shifted = labels.masked_fill(labels == -100, self.processor.tokenizer.pad_token_id)
        bos_col = torch.full(
            (shifted.size(0), 1), decoder_start, dtype=torch.long
        )
        batch["decoder_input_ids"] = torch.cat([bos_col, shifted[:, :-1]], dim=1)

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("Data collator ✓")


 
# ============================================================
# CELL 14 — compute_metrics
# ============================================================
def compute_metrics(pred):
    pred_ids  = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    label_ids = pred.label_ids.copy()
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
 

# ============================================================
# CELL 15 — Epoch Summary Callback
# ============================================================
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
        df.to_csv(EPOCH_SUMMARY_PATH, index=False)
 
epoch_callback = EpochSummaryCallback()
print("Callback ✓")
 
 
# ============================================================
# CELL 16 — Training Arguments
# ============================================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    # ADD inside Seq2SeqTrainingArguments:
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    dataloader_num_workers=DATA_LOADER_WORKERS,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=DATA_LOADER_WORKERS > 0,
    
    learning_rate=3e-4,
    lr_scheduler_type="cosine",        
    warmup_ratio=0.06,                  
 
    num_train_epochs=4,
    label_smoothing_factor=0.1,         
 
    fp16=USE_FP16,
    bf16=USE_BF16,
    fp16_full_eval=USE_FP16,
    bf16_full_eval=USE_BF16,
 
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=EVAL_NUM_BEAMS,
 
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
 
    logging_steps=10,
    logging_first_step=True,
    save_total_limit=2,
    max_grad_norm=1.0,
    report_to="none",
)
print("Training args ✓")
 
# ============================================================
# CELL 17 — Trainer + Train
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[epoch_callback],
)

print("Starting training…\n")
trainer.train()
print("\nTraining complete ✓")
