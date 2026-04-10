"""
pipeline.py
-----------
Inference and evaluation wrapper around InterConnectionST.

Usage
-----
    from pipeline import SpeechTranslationPipeline

    pipe = SpeechTranslationPipeline.from_checkpoint(
        checkpoint_path  = "./checkpoints/best_model.pt",
        asr_url          = "Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60",
        nllb_url         = "https://filesender.cesnet.cz/download.php?token=TOKEN_B&files_ids=FILE_ID",
        aggregation_layers = [6, 8, 10, 12],
        cache_dir        = "./model_cache",
    )

    result = pipe.translate_file("audio.wav")
    print(result["translation"])

    metrics = pipe.evaluate(eval_loader)
    bleu = metrics["bleu"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
from sacrebleu.metrics import BLEU, CHRF

from model_loading import (
    InterConnectionST,
    build_model,
    build_processor_and_tokenizer,
    TGT_LANG,
    NLLB_HUB,
)

SAMPLING_RATE = 16_000


def _resolve_lang_token_id(tokenizer, lang_code: str) -> int:
    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and lang_code in lang_code_to_id:
        return int(lang_code_to_id[lang_code])

    get_lang_id = getattr(tokenizer, "get_lang_id", None)
    if callable(get_lang_id):
        try:
            return int(get_lang_id(lang_code))
        except Exception:
            pass

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        token_id = convert_tokens_to_ids(lang_code)
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            return token_id

    added_tokens_encoder = getattr(tokenizer, "added_tokens_encoder", None)
    if isinstance(added_tokens_encoder, dict) and lang_code in added_tokens_encoder:
        return int(added_tokens_encoder[lang_code])

    raise AttributeError(
        f"Could not resolve language token id for '{lang_code}' "
        f"from tokenizer type {type(tokenizer).__name__}"
    )


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
        self.forced_bos_token_id = _resolve_lang_token_id(tokenizer, TGT_LANG)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_scratch(
        cls,
        asr_url: str,
        nllb_url: str,
        aggregation_layers: List[int],
        cache_dir: str = "./model_cache",
        adapter_type: str = "length",
        adapter_stride: int = 2,
        adapter_num_convs: int = 2,
        m_adapter_layers: int = 2,
        m_adapter_heads: int = 8,
        m_adapter_dropout: float = 0.1,
        decoder_lora: bool = False,
        device: str = "cuda",
        **kwargs,
    ) -> "SpeechTranslationPipeline":
        processor, tokenizer = build_processor_and_tokenizer(
            asr_url, nllb_url, cache_dir
        )
        model = build_model(
            asr_url=asr_url, nllb_url=nllb_url,
            aggregation_layers=aggregation_layers, cache_dir=cache_dir,
            adapter_type=adapter_type,
            adapter_stride=adapter_stride, adapter_num_convs=adapter_num_convs,
            m_adapter_layers=m_adapter_layers,
            m_adapter_heads=m_adapter_heads,
            m_adapter_dropout=m_adapter_dropout,
            decoder_lora=decoder_lora,
        )
        return cls(model, processor, tokenizer, device=device, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        asr_url: str,
        nllb_url: str,
        aggregation_layers: List[int],
        cache_dir: str = "./model_cache",
        adapter_type: str = "length",
        adapter_stride: int = 2,
        adapter_num_convs: int = 2,
        m_adapter_layers: int = 2,
        m_adapter_heads: int = 8,
        m_adapter_dropout: float = 0.1,
        decoder_lora: bool = False,
        device: str = "cuda",
        **kwargs,
    ) -> "SpeechTranslationPipeline":
        processor, tokenizer = build_processor_and_tokenizer(
            asr_url, nllb_url, cache_dir
        )
        model = build_model(
            asr_url=asr_url, nllb_url=nllb_url,
            aggregation_layers=aggregation_layers, cache_dir=cache_dir,
            adapter_type=adapter_type,
            adapter_stride=adapter_stride, adapter_num_convs=adapter_num_convs,
            m_adapter_layers=m_adapter_layers,
            m_adapter_heads=m_adapter_heads,
            m_adapter_dropout=m_adapter_dropout,
            decoder_lora=decoder_lora,
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

        input_name = getattr(self.processor, "_asr_input_name", "input_values")
        processor_kwargs = {
            "sampling_rate": SAMPLING_RATE,
            "return_tensors": "pt",
            "padding": True,
        }
        if input_name == "input_features":
            processor_kwargs["padding"] = "max_length"
            processor_kwargs["return_attention_mask"] = True

        enc = self.processor(waveform.astype(np.float32), **processor_kwargs)
        encoder_inputs = getattr(enc, input_name).to(self.device)
        attention_mask = (
            enc.attention_mask.to(self.device)
            if hasattr(enc, "attention_mask") and enc.attention_mask is not None
            else None
        )

        token_ids   = self.model.generate(
            encoder_inputs=encoder_inputs, attention_mask=attention_mask,
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
        encoder_inputs = batch["encoder_inputs"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_ids = self.model.generate(
            encoder_inputs=encoder_inputs, attention_mask=attention_mask,
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
    ) -> Dict[str, float]:
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

        bleu = BLEU(tokenize="13a")
        chrfpp = CHRF(word_order=2)
        bleu_result = bleu.corpus_score(hypotheses, [references])
        chrfpp_result = chrfpp.corpus_score(hypotheses, [references])
        print(f"[pipeline] Validation BLEU = {bleu_result.score:.2f}")
        print(f"[pipeline] Validation chrF++ = {chrfpp_result.score:.2f}")
        return {
            "bleu": bleu_result.score,
            "chrfpp": chrfpp_result.score,
        }

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
