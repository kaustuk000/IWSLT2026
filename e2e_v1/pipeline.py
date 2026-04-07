"""
pipeline.py
-----------
Inference and evaluation wrapper around InterConnectionST.

Usage
-----
    from pipeline import SpeechTranslationPipeline

    pipe = SpeechTranslationPipeline.from_checkpoint(
        checkpoint_path  = "./checkpoints/best_model.pt",
        wav2vec2_url     = "Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60",
        nllb_url         = "https://filesender.cesnet.cz/download.php?token=TOKEN_B&files_ids=FILE_ID",
        aggregation_layers = [6, 8, 10, 12],
        cache_dir        = "./model_cache",
    )

    result = pipe.translate_file("audio.wav")
    print(result["translation"])

    bleu = pipe.evaluate(eval_loader)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
from sacrebleu.metrics import BLEU

from model_loading import (
    InterConnectionST,
    build_model,
    build_processor_and_tokenizer,
    TGT_LANG,
    WAV2VEC2_HUB,
    NLLB_HUB,
)

SAMPLING_RATE = 16_000


class SpeechTranslationPipeline:

    def __init__(
        self,
        model: InterConnectionST,
        processor,
        tokenizer,
        device: Union[str, torch.device] = "cuda",
        num_beams: int = 4,
        max_new_tokens: int = 256,
    ):
        self.model          = model.to(device)
        self.model.eval()
        self.processor      = processor
        self.tokenizer      = tokenizer
        self.device         = torch.device(device)
        self.num_beams      = num_beams
        self.max_new_tokens = max_new_tokens
        self.forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_scratch(
        cls,
        wav2vec2_url: str,
        nllb_url: str,
        aggregation_layers: List[int],
        cache_dir: str = "./model_cache",
        adapter_stride: int = 2,
        adapter_num_convs: int = 2,
        device: str = "cuda",
        **kwargs,
    ) -> "SpeechTranslationPipeline":
        processor, tokenizer = build_processor_and_tokenizer(
            wav2vec2_url, nllb_url, cache_dir
        )
        model = build_model(
            wav2vec2_url=wav2vec2_url, nllb_url=nllb_url,
            aggregation_layers=aggregation_layers, cache_dir=cache_dir,
            adapter_stride=adapter_stride, adapter_num_convs=adapter_num_convs,
        )
        return cls(model, processor, tokenizer, device=device, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        wav2vec2_url: str,
        nllb_url: str,
        aggregation_layers: List[int],
        cache_dir: str = "./model_cache",
        adapter_stride: int = 2,
        adapter_num_convs: int = 2,
        device: str = "cuda",
        **kwargs,
    ) -> "SpeechTranslationPipeline":
        processor, tokenizer = build_processor_and_tokenizer(
            wav2vec2_url, nllb_url, cache_dir
        )
        model = build_model(
            wav2vec2_url=wav2vec2_url, nllb_url=nllb_url,
            aggregation_layers=aggregation_layers, cache_dir=cache_dir,
            adapter_stride=adapter_stride, adapter_num_convs=adapter_num_convs,
        )
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"[pipeline] Loaded weights from {ckpt}")
        else:
            print(f"[pipeline] Warning: {ckpt} not found — using base weights.")
        return cls(model, processor, tokenizer, device=device, **kwargs)

    # ------------------------------------------------------------------
    # Translate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def translate_waveform(self, waveform: np.ndarray, sample_rate: int = SAMPLING_RATE) -> Dict:
        if sample_rate != SAMPLING_RATE:
            t = torch.from_numpy(waveform).unsqueeze(0)
            t = torchaudio.functional.resample(t, sample_rate, SAMPLING_RATE)
            waveform = t.squeeze(0).numpy()

        enc            = self.processor(waveform.astype(np.float32),
                                        sampling_rate=SAMPLING_RATE,
                                        return_tensors="pt", padding=True)
        input_values   = enc.input_values.to(self.device)
        attention_mask = (enc.attention_mask.to(self.device)
                          if hasattr(enc, "attention_mask")
                          else torch.ones_like(input_values, dtype=torch.long))

        token_ids   = self.model.generate(
            input_values=input_values, attention_mask=attention_mask,
            forced_bos_token_id=self.forced_bos_token_id,
            max_new_tokens=self.max_new_tokens, num_beams=self.num_beams,
        )
        translation = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        return {"translation": translation, "token_ids": token_ids[0].tolist()}

    def translate_file(self, audio_path: str) -> Dict:
        waveform, sr = torchaudio.load(audio_path)
        waveform     = waveform.mean(dim=0).numpy()
        return self.translate_waveform(waveform, sample_rate=sr)

    @torch.no_grad()
    def translate_batch(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        input_values   = batch["input_values"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_ids = self.model.generate(
            input_values=input_values, attention_mask=attention_mask,
            forced_bos_token_id=self.forced_bos_token_id,
            max_new_tokens=self.max_new_tokens, num_beams=self.num_beams,
        )
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        print_examples: int = 3,
    ) -> float:
        self.model.eval()
        hypotheses, references = [], []

        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            preds = self.translate_batch(batch)
            hypotheses.extend(preds)
            lids = batch["labels"].clone()
            lids[lids == -100] = self.tokenizer.pad_token_id
            refs = self.tokenizer.batch_decode(lids, skip_special_tokens=True)
            references.extend(refs)

        if print_examples:
            print("\n[pipeline] Examples:")
            for i in range(min(print_examples, len(hypotheses))):
                print(f"  REF: {references[i]}")
                print(f"  HYP: {hypotheses[i]}\n")

        bleu   = BLEU(tokenize="char")
        result = bleu.corpus_score(hypotheses, [references])
        print(f"[pipeline] BLEU = {result.score:.2f}")
        return result.score

    # ------------------------------------------------------------------
    # Inspect aggregation weights
    # ------------------------------------------------------------------

    def get_layer_weights(self) -> Dict[int, float]:
        import torch.nn.functional as F
        weights = F.softmax(
            self.model.aggregator.layer_weights.detach().cpu(), dim=0
        ).tolist()
        return dict(zip(self.model.aggregator.layer_indices, weights))

    def print_layer_weights(self):
        import torch.nn.functional as F
        weights = F.softmax(
            self.model.aggregator.layer_weights.detach().cpu(), dim=0
        ).tolist()
        print("\n[pipeline] Aggregation layer weights:")
        for idx, w in zip(self.model.aggregator.layer_indices, weights):
            bar = "█" * int(w * 50)
            print(f"  Layer {idx:3d}: {w:.4f}  {bar}")
