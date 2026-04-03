import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from dataclasses import dataclass


class BhoHinDataset(Dataset):
    def __init__(self, base_path, split="train"):
        self.base_path = base_path
        self.split = split
        self.df = pd.read_csv(
            f"{base_path}/{split}/stamped.tsv",
            sep="\t", header=None,
            names=["wav_path", "start", "duration"],
        )
        with open(f"{base_path}/{split}/txt/{split}.hi", "r", encoding="utf-8") as f:
            self.hindi = [l.strip() for l in f.readlines()]
        assert len(self.df) == len(self.hindi)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = f"{self.base_path}/{self.split}/{row.wav_path}"
        info = torchaudio.info(audio_path)
        frame_offset = max(0, int(round(float(row.start) * info.sample_rate)))
        num_frames = max(1, int(round(float(row.duration) * info.sample_rate)))
        waveform, sr = torchaudio.load(
            audio_path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return {"audio": waveform.squeeze(0), "sampling_rate": sr, "hindi": self.hindi[idx]}


@dataclass
class BhoHinCollator:
    processor: any
    tokenizer: any
    max_label_len: int = 128

    def __call__(self, batch):
        target_sr = self.processor.feature_extractor.sampling_rate
        audio_arrays = []
        for item in batch:
            audio = item["audio"]
            if item["sampling_rate"] != target_sr:
                audio = torchaudio.functional.resample(audio, item["sampling_rate"], target_sr)
            audio_arrays.append(audio.numpy())
        hindi_texts = [b["hindi"] for b in batch]
        audio_inputs = self.processor(
            audio_arrays, sampling_rate=target_sr,
            return_tensors="pt", padding=True,
        )
        label_enc = self.tokenizer(
            hindi_texts, return_tensors="pt",
            padding=True, truncation=True,
            max_length=self.max_label_len, add_special_tokens=True,
        )
        labels = label_enc.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_values": audio_inputs.input_values,
            "attention_mask": audio_inputs.attention_mask,
            "labels": labels,
        }


def make_collate_fn(processor, tokenizer, max_label_len=128):
    return BhoHinCollator(processor=processor, tokenizer=tokenizer, max_label_len=max_label_len)
