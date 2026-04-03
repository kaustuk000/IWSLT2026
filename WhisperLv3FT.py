!pip install -q transformers datasets accelerate peft jiwer librosa audiomentations
 
print("done")


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




import os, gc, math, json, itertools, subprocess, tempfile
import torch
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
MAX_AUDIO_SEC = 20          # ← reduced from 30 (saves memory)
SEEDS         = [42, 1337, 2024]
OUTPUT_DIR    = "/kaggle/working/whisper-bhojpuri"
SAVE_PATH     = "/kaggle/working/whisper-bhojpuri-final"
LM_ARPA_PATH  = "/kaggle/working/bhojpuri_lm.arpa"

SPLIT_REAL      = "train_real"
SPLIT_SYNTHETIC = "train_synthetic"
SPLIT_BENCHMARK = "benchmark"

# ── Tiny sizes for quick 1-epoch smoke test ──
REAL_SAMPLES      = 20
SYNTHETIC_SAMPLES = 80
TEST_SIZE         = 40
VAL_SIZE          = 40
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
real_samples = list(ds_real.take(REAL_SAMPLES))   # ← capped
print(f"  train_real      : {len(real_samples)} samples")

print("Loading train_synthetic…")
ds_syn = load_dataset(DATASET_NAME, split=SPLIT_SYNTHETIC, streaming=True)
syn_samples = list(ds_syn.take(SYNTHETIC_SAMPLES))
print(f"  train_synthetic : {len(syn_samples)} samples")

print("Loading benchmark…")
ds_bench = load_dataset(DATASET_NAME, split=SPLIT_BENCHMARK, streaming=True)
bench_samples = list(ds_bench.take(VAL_SIZE + TEST_SIZE))   # ← only take what we need
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