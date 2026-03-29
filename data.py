import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


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
        waveform, sr = torchaudio.load(f"{self.base_path}/{self.split}/{row.wav_path}")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return {"audio": waveform.squeeze(0), "sampling_rate": sr, "hindi": self.hindi[idx]}


def make_collate_fn(processor, tokenizer, max_label_len=128):
    def collate_fn(batch):
        audio_arrays = [b["audio"].numpy() for b in batch]
        hindi_texts  = [b["hindi"] for b in batch]
        audio_inputs = processor(
            audio_arrays, sampling_rate=16000,
            return_tensors="pt", padding=True,
        )
        label_enc = tokenizer(
            hindi_texts, return_tensors="pt",
            padding=True, truncation=True,
            max_length=max_label_len, add_special_tokens=True,
        )
        labels = label_enc.input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_values":   audio_inputs.input_values,
            "attention_mask": audio_inputs.attention_mask,
            "labels":         labels,
        }
    return collate_fn
