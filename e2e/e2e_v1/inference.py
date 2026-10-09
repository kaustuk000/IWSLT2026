"""
inference.py
------------
Batch inference entrypoint for the Bhojpuri -> Hindi ST model.

Expected test split layout:
    <test_dir>/
        stamped.tsv
        wav/
          ...

Unlike training/dev splits, no txt/ folder is required.

Example:
    python inference.py \
        --checkpoint_path ./checkpoints/best_model.pt \
        --asr_url Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60 \
        --nllb_url ./model_cache/nllb/model-mt-v1_best \
        --test_dir ./data/test \
        --output_dir ./test_predictions
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from data_loading import SAMPLING_RATE
from model_loading import build_model, build_processor_and_tokenizer
from pipeline import SpeechTranslationPipeline


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to the saved checkpoint, e.g. best_model.pt")
    parser.add_argument("--config_json", default=None,
                        help="Optional training config JSON. If omitted, inference will look for run_config.json next to the checkpoint.")
    parser.add_argument("--asr_url", "--wav2vec2_url", dest="asr_url", default=None,
                        help="Hugging Face repo/URL, FileSender URL, or local path/tarball for the ASR encoder")
    parser.add_argument("--nllb_url", default=None,
                        help="Direct FileSender file URL or local path/tarball for NLLB")
    parser.add_argument("--cache_dir", default="./model_cache",
                        help="Local directory to cache downloaded/extracted models")

    parser.add_argument("--test_dir", required=True,
                        help="Directory containing stamped.tsv and the audio tree for test inference")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where predictions are written")

    parser.add_argument("--aggregation_layers", nargs="+", type=int,
                        default=[6, 8, 10, 12],
                        help="ASR encoder layer indices to aggregate. Pass 0 alone to use ALL layers.")
    parser.add_argument("--adapter_type", choices=["length", "m_adapter"], default="length",
                        help="Bridge from ASR encoder to NLLB decoder")
    parser.add_argument("--adapter_stride", type=int, default=2)
    parser.add_argument("--adapter_num_convs", type=int, default=2)
    parser.add_argument("--m_adapter_layers", type=int, default=2)
    parser.add_argument("--m_adapter_heads", type=int, default=8)
    parser.add_argument("--m_adapter_dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze_decoder_lora", action="store_true",
                        help="Attach LoRA adapters when loading a LoRA-trained checkpoint")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_beams", type=int, default=4)
    parser.add_argument("--eval_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--eval_no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    return parser


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_config_path(args) -> Path | None:
    if args.config_json:
        config_path = Path(args.config_json).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config_path

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    candidates = [
        checkpoint_path.parent / "run_config.json",
        checkpoint_path.with_suffix(".json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _apply_config_defaults(args, parser, config: dict):
    for action in parser._actions:
        dest = action.dest
        if dest in {"help", "config_json"}:
            continue
        if dest not in config:
            continue
        current_value = getattr(args, dest, None)
        default_value = parser.get_default(dest)
        if current_value is None or current_value == default_value:
            setattr(args, dest, config[dest])
    return args


def _validate_required_runtime_args(args):
    missing = []
    for field in ("asr_url", "nllb_url"):
        if not getattr(args, field, None):
            missing.append(f"--{field}")
    if missing:
        raise ValueError(
            "Inference still needs model source information. Missing: "
            + ", ".join(missing)
            + ". Pass them directly or provide a run_config.json from training."
        )


def parse_args():
    parser = build_parser()
    args = parser.parse_args()

    config_path = _resolve_config_path(args)
    if config_path is not None:
        config = _load_json(config_path)
        args = _apply_config_defaults(args, parser, config)
        args.config_json = str(config_path)
        print(f"[inference] Loaded config defaults from: {config_path}")

    _validate_required_runtime_args(args)
    return args


def _resolve_manifest_path(test_dir: Path) -> Path:
    candidates = [
        test_dir / "stamped.tsv",
        test_dir / "stamped.tsc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find stamped.tsv under {test_dir}. "
        f"Checked: {[str(p) for p in candidates]}"
    )


class TestInferenceDataset(Dataset):
    def __init__(self, test_dir: str, processor):
        self.test_dir = Path(test_dir).expanduser().resolve()
        self.processor = processor
        self.manifest_path = _resolve_manifest_path(self.test_dir)
        self.df = pd.read_csv(
            self.manifest_path,
            sep="\t",
            header=None,
            names=["wav_path", "start", "duration"],
        )
        self.input_name = getattr(self.processor, "_asr_input_name", "input_values")
        print(f"[inference] Loaded {len(self.df)} test items from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        audio_path = self.test_dir / str(row.wav_path)
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != SAMPLING_RATE:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sr, SAMPLING_RATE
            ).squeeze(0)

        waveform = waveform.numpy().astype(np.float32)

        processor_kwargs = {
            "sampling_rate": SAMPLING_RATE,
            "return_tensors": "np",
            "padding": False,
        }
        if self.input_name == "input_features":
            processor_kwargs["padding"] = "max_length"
            processor_kwargs["return_attention_mask"] = True

        enc = self.processor(waveform, **processor_kwargs)
        encoder_inputs = getattr(enc, self.input_name)[0]
        attention_mask = (
            enc.attention_mask[0]
            if hasattr(enc, "attention_mask") and enc.attention_mask is not None
            else np.ones(encoder_inputs.shape[-1], dtype=np.int32)
        )

        return {
            "index": idx,
            "wav_path": str(row.wav_path),
            "start": float(row.start),
            "duration": float(row.duration),
            "encoder_inputs": encoder_inputs,
            "attention_mask": attention_mask,
        }


class InferenceCollator:
    def __call__(self, features: List[Dict]) -> Dict:
        input_lengths = [f["encoder_inputs"].shape[-1] for f in features]
        max_audio = max(input_lengths)
        sample = features[0]["encoder_inputs"]

        if sample.ndim == 1:
            encoder_inputs = np.zeros((len(features), max_audio), dtype=np.float32)
            for i, feature in enumerate(features):
                length = input_lengths[i]
                encoder_inputs[i, :length] = feature["encoder_inputs"]
        elif sample.ndim == 2:
            feature_dim = sample.shape[0]
            encoder_inputs = np.zeros((len(features), feature_dim, max_audio), dtype=np.float32)
            for i, feature in enumerate(features):
                length = input_lengths[i]
                encoder_inputs[i, :, :length] = feature["encoder_inputs"][:, :length]
        else:
            raise ValueError(f"Unsupported encoder input rank: {sample.ndim}")

        attention_mask = np.zeros((len(features), max_audio), dtype=np.int64)
        for i, feature in enumerate(features):
            length = input_lengths[i]
            attention_mask[i, :length] = feature["attention_mask"][:length]

        return {
            "indices": [int(f["index"]) for f in features],
            "wav_paths": [f["wav_path"] for f in features],
            "starts": [float(f["start"]) for f in features],
            "durations": [float(f["duration"]) for f in features],
            "encoder_inputs": torch.tensor(encoder_inputs),
            "attention_mask": torch.tensor(attention_mask),
        }


def _effective_num_workers(num_workers: int) -> int:
    if hasattr(os, "sched_getaffinity"):
        available_workers = max(1, len(os.sched_getaffinity(0)))
    else:
        available_workers = max(1, os.cpu_count() or 1)
    effective = min(num_workers, available_workers)
    if effective != num_workers:
        print(
            f"[inference] Reducing num_workers from {num_workers} to {effective} "
            "based on available CPU workers."
        )
    return effective


def _build_inference_loader(test_dir: str, processor, batch_size: int, num_workers: int):
    dataset = TestInferenceDataset(test_dir, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_effective_num_workers(num_workers),
        collate_fn=InferenceCollator(),
        pin_memory=True,
    )
    return dataset, loader


def _load_pipeline(args, device: torch.device) -> SpeechTranslationPipeline:
    processor, tokenizer = build_processor_and_tokenizer(
        args.asr_url, args.nllb_url, args.cache_dir
    )
    aggregation_layers = [] if args.aggregation_layers == [0] else args.aggregation_layers
    model = build_model(
        asr_url=args.asr_url,
        nllb_url=args.nllb_url,
        aggregation_layers=aggregation_layers,
        cache_dir=args.cache_dir,
        adapter_type=args.adapter_type,
        adapter_stride=args.adapter_stride,
        adapter_num_convs=args.adapter_num_convs,
        m_adapter_layers=args.m_adapter_layers,
        m_adapter_heads=args.m_adapter_heads,
        m_adapter_dropout=args.m_adapter_dropout,
        freeze_encoder=True,
        freeze_decoder=True,
        decoder_lora=args.unfreeze_decoder_lora,
    )

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[inference] Loaded checkpoint from: {checkpoint_path}")
    if missing:
        print(f"[inference] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[inference] Unexpected keys: {len(unexpected)}")

    return SpeechTranslationPipeline(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device,
        num_beams=args.eval_beams,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.eval_repetition_penalty,
        no_repeat_ngram_size=args.eval_no_repeat_ngram_size,
    )


def _save_predictions(output_dir: Path, dataset: TestInferenceDataset, rows: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(rows, key=lambda row: row["index"])
    predictions = [row["translation"] for row in rows]

    predictions_text = "\n".join(predictions) + "\n"
    predictions_txt = txt_dir / "predictions.hi"
    test_txt = txt_dir / "test.hi"
    predictions_txt.write_text(predictions_text, encoding="utf-8")
    test_txt.write_text(predictions_text, encoding="utf-8")

    predictions_df = pd.DataFrame(
        [
            {
                "wav_path": row["wav_path"],
                "start": row["start"],
                "duration": row["duration"],
                "translation": row["translation"],
            }
            for row in rows
        ]
    )
    predictions_df.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)

    shutil.copy2(dataset.manifest_path, output_dir / "stamped.tsv")
    print(f"[inference] Saved translations to: {predictions_txt}")
    print(f"[inference] Saved dataset-style text file to: {test_txt}")
    print(f"[inference] Saved structured predictions to: {output_dir / 'predictions.tsv'}")


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Device: {device}")

    pipe = _load_pipeline(args, device)
    dataset, loader = _build_inference_loader(
        args.test_dir,
        pipe.processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = []
    for batch_idx, batch in enumerate(loader, start=1):
        predictions = pipe.translate_batch(batch)
        for index, wav_path, start, duration, prediction in zip(
            batch["indices"],
            batch["wav_paths"],
            batch["starts"],
            batch["durations"],
            predictions,
        ):
            results.append(
                {
                    "index": index,
                    "wav_path": wav_path,
                    "start": start,
                    "duration": duration,
                    "translation": prediction,
                }
            )

        if batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"[inference] Processed {len(results)}/{len(dataset)} items")

    _save_predictions(Path(args.output_dir).expanduser().resolve(), dataset, results)
    print("[inference] Done.")


if __name__ == "__main__":
    run_inference(parse_args())
