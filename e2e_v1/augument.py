"""
augument.py
-----------
Offline dataset augmentation for the Bhojpuri-Hindi ST data.

This script:
1. Locates the dataset root using the same logic as the training pipeline.
2. Creates spectrogram-masked waveform variants for a split (default: train).
3. Appends the augmented examples to the split manifest and text file.
4. Shuffles the combined original + augmented examples together.

The script is idempotent with respect to manifest growth: it keeps backups of the
original manifest/text and always rebuilds the augmented split from those backups.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torchaudio

from data_loading import DEFAULT_DATA_DIR, SAMPLING_RATE, ensure_dataset_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Dataset root or clone target. If missing, the repo at "
                             "https://github.com/shashwatup9k/iwslt2026_bho-hi is downloaded.")
    parser.add_argument("--split", default="train",
                        help="Split to augment (default: train)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_subdir", default="wav_aug_specaug",
                        help="Relative subdir under the split where augmented wavs are written")
    parser.add_argument("--suffix", default="specaug",
                        help="Suffix appended to augmented audio filenames")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--win_length", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--griffin_lim_iters", type=int, default=32)
    parser.add_argument("--time_mask_max", type=int, default=30,
                        help="Maximum masked time frames")
    parser.add_argument("--freq_mask_max", type=int, default=30,
                        help="Maximum masked mel bins")
    parser.add_argument("--num_time_masks", type=int, default=1)
    parser.add_argument("--num_freq_masks", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Optional cap for quick experiments")
    return parser.parse_args()


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".orig")


def _ensure_backups(tsv_path: Path, text_path: Path) -> tuple[Path, Path]:
    tsv_backup = _backup_path(tsv_path)
    text_backup = _backup_path(text_path)
    if not tsv_backup.exists():
        tsv_backup.write_text(tsv_path.read_text(encoding="utf-8"), encoding="utf-8")
    if not text_backup.exists():
        text_backup.write_text(text_path.read_text(encoding="utf-8"), encoding="utf-8")
    return tsv_backup, text_backup


def _load_base_split(tsv_path: Path, text_path: Path) -> tuple[pd.DataFrame, List[str]]:
    tsv_backup, text_backup = _ensure_backups(tsv_path, text_path)
    df = pd.read_csv(
        tsv_backup,
        sep="\t",
        header=None,
        names=["wav_path", "start", "duration"],
    )
    lines = text_backup.read_text(encoding="utf-8").splitlines()
    if len(df) != len(lines):
        raise ValueError(
            f"Manifest/text length mismatch: {len(df)} rows vs {len(lines)} lines "
            f"for {tsv_backup} and {text_backup}"
        )
    return df, lines


def _build_specaugment_transforms(args):
    spec = torchaudio.transforms.Spectrogram(
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        power=2.0,
    )
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_iter=args.griffin_lim_iters,
        power=2.0,
    )
    return spec, griffin_lim


def _apply_specaugment(
    waveform: torch.Tensor,
    *,
    spec_transform,
    griffin_lim_transform,
    time_mask_max: int,
    freq_mask_max: int,
    num_time_masks: int,
    num_freq_masks: int,
) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Use a linear-frequency spectrogram here rather than a mel spectrogram.
    # Inverting masked mel features via least-squares was numerically unstable
    # on some utterances; masking the linear spectrogram keeps the same spirit
    # of SpecAugment while remaining robust for offline waveform synthesis.
    spec = spec_transform(waveform)
    if spec.ndim == 2:
        spec = spec.unsqueeze(0)
    masked = spec.clone()

    for _ in range(num_freq_masks):
        width = random.randint(0, min(freq_mask_max, masked.shape[-2]))
        if width <= 0:
            continue
        start = random.randint(0, masked.shape[-2] - width)
        masked[:, start:start + width, :] = 0

    for _ in range(num_time_masks):
        width = random.randint(0, min(time_mask_max, masked.shape[-1]))
        if width <= 0:
            continue
        start = random.randint(0, masked.shape[-1] - width)
        masked[:, :, start:start + width] = 0

    augmented = griffin_lim_transform(masked.clamp_min(1e-10))
    if augmented.ndim == 1:
        augmented = augmented.unsqueeze(0)
    return augmented


def _prepare_waveform(audio_path: Path) -> tuple[torch.Tensor, float]:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLING_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLING_RATE)
        sample_rate = SAMPLING_RATE
    duration = waveform.shape[-1] / sample_rate
    return waveform, duration


def _write_augmented_split(
    split_dir: Path,
    df: pd.DataFrame,
    text_lines: List[str],
    args,
) -> Tuple[int, int]:
    output_dir = split_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_transform, griffin_lim = _build_specaugment_transforms(args)

    rows = df.to_dict("records")
    limit = args.max_examples if args.max_examples is not None else len(rows)
    new_rows = []
    new_lines = []

    for idx, (row, text) in enumerate(zip(rows[:limit], text_lines[:limit])):
        src_rel = Path(str(row["wav_path"]))
        src_audio = split_dir / src_rel
        waveform, duration = _prepare_waveform(src_audio)
        augmented = _apply_specaugment(
            waveform,
            spec_transform=spec_transform,
            griffin_lim_transform=griffin_lim,
            time_mask_max=args.time_mask_max,
            freq_mask_max=args.freq_mask_max,
            num_time_masks=args.num_time_masks,
            num_freq_masks=args.num_freq_masks,
        )

        aug_name = f"{src_rel.stem}__{args.suffix}_{idx:06d}.wav"
        aug_rel = Path(args.output_subdir) / aug_name
        aug_path = split_dir / aug_rel
        torchaudio.save(str(aug_path), augmented.cpu(), SAMPLING_RATE)

        new_rows.append(
            {
                "wav_path": aug_rel.as_posix(),
                "start": 0.0,
                "duration": duration,
            }
        )
        new_lines.append(text)

    combined_rows = rows + new_rows
    combined_lines = text_lines + new_lines
    combined = list(zip(combined_rows, combined_lines))
    random.shuffle(combined)

    shuffled_rows = [item[0] for item in combined]
    shuffled_lines = [item[1] for item in combined]

    out_df = pd.DataFrame(shuffled_rows, columns=["wav_path", "start", "duration"])
    (split_dir / "stamped.tsv").write_text(
        out_df.to_csv(sep="\t", header=False, index=False, lineterminator="\n"),
        encoding="utf-8",
    )
    (split_dir / "txt" / f"{args.split}.hi").write_text(
        "\n".join(shuffled_lines) + "\n",
        encoding="utf-8",
    )
    return len(rows), len(new_rows)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(ensure_dataset_root(args.data_dir, train_split=args.split))
    split_dir = dataset_root / args.split
    tsv_path = split_dir / "stamped.tsv"
    text_path = split_dir / "txt" / f"{args.split}.hi"

    if not tsv_path.exists() or not text_path.exists():
        raise FileNotFoundError(
            f"Expected split files at {tsv_path} and {text_path}"
        )

    df, text_lines = _load_base_split(tsv_path, text_path)
    base_count, aug_count = _write_augmented_split(split_dir, df, text_lines, args)
    print(
        f"[augument] Rebuilt split '{args.split}' from {base_count} original examples "
        f"plus {aug_count} SpecAugment examples."
    )
    print(f"[augument] Updated manifest: {tsv_path}")
    print(f"[augument] Updated text file: {text_path}")


if __name__ == "__main__":
    main()
