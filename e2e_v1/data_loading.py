"""
data_loading.py
---------------
Data loading for the IWSLT2026 Bhojpuri-Hindi ST dataset.

Dataset layout (per split):
    {base_path}/
        {split}/
            stamped.tsv          ← wav_path <TAB> start <TAB> duration  (no header)
            txt/
                {split}.hi       ← one Hindi translation per line (aligned to TSV)
            {wav_path}           ← audio files (wav / any torchaudio-readable format)
"""

from __future__ import annotations

import random
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve
import zipfile

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor

SAMPLING_RATE     = 16_000
MAX_AUDIO_LEN_SEC = 30
MAX_LABEL_TOKENS  = 128
DATASET_REPO_URL = "https://github.com/shashwatup9k/iwslt2026_bho-hi"
DATASET_REPO_ZIP_URL = f"{DATASET_REPO_URL}/archive/refs/heads/main.zip"
DATASET_REPO_SUBDIR = "iwslt2024-2025_bho-hi"


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def speed_perturb(waveform: np.ndarray, sr: int, factors=(0.9, 1.0, 1.1)) -> np.ndarray:
    factor = random.choice(factors)
    if factor == 1.0:
        return waveform
    tensor   = torch.from_numpy(waveform).unsqueeze(0)
    new_sr   = int(sr * factor)
    resampled = torchaudio.functional.resample(tensor, new_sr, sr)
    return resampled.squeeze(0).numpy()


def add_gaussian_noise(waveform: np.ndarray, snr_db_range=(15, 40)) -> np.ndarray:
    snr_db       = random.uniform(*snr_db_range)
    signal_power = np.mean(waveform ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    return (waveform + noise).astype(np.float32)


def spec_augment(
    input_values: np.ndarray,
    time_mask_param: int = 100,
    num_time_masks:  int = 2,
) -> np.ndarray:
    arr    = input_values.copy()
    length = arr.shape[-1]
    for _ in range(num_time_masks):
        t  = random.randint(0, min(time_mask_param, length - 1))
        t0 = random.randint(0, max(0, length - t))
        arr[..., t0 : t0 + t] = 0.0
    return arr


# ---------------------------------------------------------------------------
# Dataset discovery / download helpers
# ---------------------------------------------------------------------------

def _split_manifest_exists(base_path: Path, split: str) -> bool:
    return (base_path / split / "stamped.tsv").exists()


def _dataset_root_candidates(base_path: Path) -> List[Path]:
    return [
        base_path,
        base_path / DATASET_REPO_SUBDIR,
    ]


def _find_existing_dataset_root(base_path: Path, train_split: str) -> Optional[Path]:
    for candidate in _dataset_root_candidates(base_path):
        if _split_manifest_exists(candidate, train_split):
            return candidate
    return None


def _download_dataset_repo(base_path: Path) -> None:
    """
    Download the dataset repo into `base_path`.

    Prefers `git clone` when available and falls back to downloading the
    GitHub main branch zip archive.
    """
    parent = base_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    git_exe = shutil.which("git")
    if git_exe:
        print(f"[data_loading] Cloning dataset repo into: {base_path}")
        try:
            subprocess.run(
                [git_exe, "clone", "--depth", "1", DATASET_REPO_URL, str(base_path)],
                check=True,
            )
            return
        except subprocess.CalledProcessError as exc:
            print(f"[data_loading] git clone failed ({exc}); falling back to zip download.")

    print(f"[data_loading] Downloading dataset archive from: {DATASET_REPO_ZIP_URL}")
    zip_path = parent / "iwslt2026_bho-hi-main.zip"
    extract_dir = parent / "iwslt2026_bho-hi-main.extract"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    try:
        urlretrieve(DATASET_REPO_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        extracted_roots = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(extracted_roots) != 1:
            raise RuntimeError(
                f"Expected one extracted root directory, found {len(extracted_roots)}."
            )

        shutil.move(str(extracted_roots[0]), str(base_path))
    finally:
        zip_path.unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


def ensure_dataset_root(
    base_path: str,
    train_split: str = "train",
) -> str:
    """
    Ensure the Bhojpuri-Hindi dataset is available locally and return the
    dataset root that contains the train/dev split folders.

    Accepted layouts:
      - {base_path}/train/...                       (already resolved root)
      - {base_path}/iwslt2024-2025_bho-hi/train/... (repo clone root)
    """
    root = Path(base_path).expanduser().resolve()

    existing_root = _find_existing_dataset_root(root, train_split)
    if existing_root is not None:
        print(f"[data_loading] Using dataset at: {existing_root}")
        return str(existing_root)

    if root.exists():
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset path exists but is not a directory: {root}")
        if any(root.iterdir()):
            raise FileNotFoundError(
                f"Dataset not found under existing directory: {root}. "
                f"Expected '{train_split}/stamped.tsv' either directly under it "
                f"or under '{DATASET_REPO_SUBDIR}/'."
            )

    print(f"[data_loading] Dataset missing at {root}; fetching from GitHub...")
    _download_dataset_repo(root)

    existing_root = _find_existing_dataset_root(root, train_split)
    if existing_root is not None:
        print(f"[data_loading] Dataset ready at: {existing_root}")
        return str(existing_root)

    raise FileNotFoundError(
        "Dataset download completed, but the expected training split was not found. "
        f"Checked '{root}' and '{root / DATASET_REPO_SUBDIR}'."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BhoHinSTDataset(Dataset):
    """
    Bhojpuri → Hindi speech translation dataset.

    Args:
        base_path         : root folder of the dataset
        split             : "train" | "dev" | "test"
        processor         : Wav2Vec2Processor
        tokenizer         : NLLB tokenizer
        augment           : apply speed perturb + noise + SpecAugment
        max_audio_len_sec : drop utterances longer than this
    """

    def __init__(
        self,
        base_path: str,
        split: str,
        processor: Wav2Vec2Processor,
        tokenizer,
        augment: bool = False,
        max_audio_len_sec: float = MAX_AUDIO_LEN_SEC,
    ):
        self.base_path   = base_path
        self.split       = split
        self.processor   = processor
        self.tokenizer   = tokenizer
        self.augment     = augment
        self.max_samples = int(max_audio_len_sec * SAMPLING_RATE)

        # ---- manifest ----
        tsv_path = f"{base_path}/{split}/stamped.tsv"
        self.df  = pd.read_csv(
            tsv_path, sep="\t", header=None,
            names=["wav_path", "start", "duration"],
        )

        # ---- hindi translations ----
        hi_path = f"{base_path}/{split}/txt/{split}.hi"
        with open(hi_path, "r", encoding="utf-8") as f:
            self.hindi = [l.strip() for l in f.readlines()]

        assert len(self.df) == len(self.hindi), (
            f"TSV has {len(self.df)} rows but {hi_path} has {len(self.hindi)} lines."
        )

        # ---- filter long utterances via duration column ----
        mask       = self.df["duration"].astype(float) <= max_audio_len_sec
        dropped    = (~mask).sum()
        self.df    = self.df[mask].reset_index(drop=True)
        self.hindi = [h for h, m in zip(self.hindi, mask) if m]
        if dropped:
            print(f"[dataset:{split}] Dropped {dropped} utterances > {max_audio_len_sec}s.")

        print(f"[dataset:{split}] {len(self.df)} examples ready.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        # ---- load audio ----
        audio_path   = f"{self.base_path}/{self.split}/{row.wav_path}"
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:                          # stereo → mono
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)                     # (T,)

        if sr != SAMPLING_RATE:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sr, SAMPLING_RATE
            ).squeeze(0)

        waveform = waveform.numpy().astype(np.float32)

        # ---- augmentation (train only) ----
        if self.augment:
            if random.random() < 0.5:
                waveform = speed_perturb(waveform, SAMPLING_RATE)
            if random.random() < 0.3:
                waveform = add_gaussian_noise(waveform)

        # ---- encoder features ----
        enc            = self.processor(
            waveform, sampling_rate=SAMPLING_RATE,
            return_tensors="np", padding=False,
        )
        input_values   = enc.input_values[0]               # (T,)
        attention_mask = (
            enc.attention_mask[0]
            if hasattr(enc, "attention_mask") and enc.attention_mask is not None
            else np.ones(len(input_values), dtype=np.int32)
        )

        if self.augment and random.random() < 0.5:
            input_values = spec_augment(input_values)

        # ---- decoder labels ----
        self.tokenizer.src_lang = "bho_Deva"
        self.tokenizer.tgt_lang = "hin_Deva"
        label_enc = self.tokenizer(
            text_target=self.hindi[idx],
            max_length=MAX_LABEL_TOKENS,
            truncation=True,
            padding=False,
        )

        return {
            "input_values":   input_values,
            "attention_mask": attention_mask,
            "labels":         label_enc["input_ids"],
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class SpeechTranslationCollator:
    """Pads to longest sample in batch. Labels padded with -100."""
    pad_token_id: int = 1     # NLLB <pad> = 1

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_lengths = [len(f["input_values"]) for f in features]
        max_aud = max(input_lengths)

        input_values   = np.zeros((len(features), max_aud), dtype=np.float32)
        attention_mask = np.zeros((len(features), max_aud), dtype=np.int64)
        for i, f in enumerate(features):
            l = input_lengths[i]
            input_values[i, :l]   = f["input_values"]
            attention_mask[i, :l] = f["attention_mask"][:l]

        label_lengths = [len(f["labels"]) for f in features]
        max_lab = max(label_lengths)
        labels  = np.full((len(features), max_lab), -100, dtype=np.int64)
        for i, f in enumerate(features):
            l = label_lengths[i]
            labels[i, :l] = f["labels"]

        return {
            "input_values":   torch.tensor(input_values),
            "attention_mask": torch.tensor(attention_mask),
            "labels":         torch.tensor(labels),
        }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def get_dataloaders(
    base_path: str,
    processor: Wav2Vec2Processor,
    tokenizer,
    batch_size:        int            = 4,
    num_workers:       int            = 4,
    train_split:       str            = "train",
    eval_split:        Optional[str]  = "dev",
    max_audio_len_sec: float          = MAX_AUDIO_LEN_SEC,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Returns (train_loader, eval_loader).
    eval_loader is None if eval_split is None or the split directory is missing.
    """
    if hasattr(os, "sched_getaffinity"):
        available_workers = max(1, len(os.sched_getaffinity(0)))
    else:
        available_workers = max(1, os.cpu_count() or 1)
    effective_num_workers = min(num_workers, available_workers)
    if effective_num_workers != num_workers:
        print(
            f"[data_loading] Reducing num_workers from {num_workers} to "
            f"{effective_num_workers} based on available CPU workers."
        )

    base_path = ensure_dataset_root(base_path, train_split=train_split)
    collator = SpeechTranslationCollator(pad_token_id=tokenizer.pad_token_id or 1)

    train_ds = BhoHinSTDataset(
        base_path=base_path, split=train_split,
        processor=processor, tokenizer=tokenizer,
        augment=True, max_audio_len_sec=max_audio_len_sec,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=effective_num_workers, collate_fn=collator, pin_memory=True,
    )

    eval_loader = None
    if eval_split is not None:
        try:
            eval_ds = BhoHinSTDataset(
                base_path=base_path, split=eval_split,
                processor=processor, tokenizer=tokenizer,
                augment=False, max_audio_len_sec=max_audio_len_sec,
            )
            eval_loader = DataLoader(
                eval_ds, batch_size=batch_size, shuffle=False,
                num_workers=effective_num_workers, collate_fn=collator, pin_memory=True,
            )
        except Exception as e:
            print(f"[data_loading] Could not load eval split '{eval_split}': {e}")

    return train_loader, eval_loader
