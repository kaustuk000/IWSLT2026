"""
inference_exp1.py
=================
Experiment 1 — 3-stage cascade inference pipeline:

  Stage 1 │ Whisper (fine-tuned)       Bhojpuri audio  →  noisy Bhojpuri text
  Stage 2 │ LLM ASR corrector          noisy Bhojpuri  →  clean Bhojpuri text
  Stage 3 │ NLLB-200 MT                clean Bhojpuri  →  final Hindi

Stage 4 (LLM MT post-editor) has been intentionally removed.
NLLB-200 output is the final translation output.

Zero-shot mode  (--zeroshot)
-----------------------------
  Sub-modes via --zeroshot_mode  (default: full):
    full       Base LLM handles Stage 2.
    skip_s2    Stage 2 skipped; raw ASR passed directly to NLLB.

QE Fusion mode  (--qe_fusion)
------------------------------
  Replaces greedy/beam-search decoding with the QE Fusion algorithm
  (Vernikos et al., 2024 — "Don't Rank, Combine!").
  Candidates are generated via sampling; QE-Kiwi selects and recombines them.

  --qe_fusion choices:
    both   (default)  Apply QE Fusion to Stage 2 (LLM) AND Stage 3 (NLLB).
    llm                Apply QE Fusion to Stage 2 (LLM) only.
    mt                 Apply QE Fusion to Stage 3 (NLLB) only.
    none               Disable QE Fusion; use original greedy/beam-search.

  Stage 2 candidate generation : nucleus sampling  p=0.9, T=0.6
  Stage 3 candidate generation : epsilon sampling  ε=0.02, T=0.5
  QE metric                    : Unbabel/wmt22-cometkiwi-da  (reference-free)

  Note on Stage 2 QE scoring:
    COMET-Kiwi was designed for MT quality estimation.  For Stage 2 (monolingual
    Bhojpuri ASR correction) we treat the noisy ASR text as the "source" and the
    LLM correction as the "hypothesis".  The model provides a useful relative
    ranking signal between candidates even in this non-standard setting.

Sample selection
----------------
  --num_samples N    Process only the first N samples. Omit for all.
"""

from __future__ import annotations
# 🔥 MUST BE FIRST THING IN FILE
import logging

def _kill_lightning_logs():
    def silent(*args, **kwargs):
        pass

    # Kill Lightning rank_zero logs
    try:
        import lightning.pytorch.utilities.rank_zero as rz
        rz.rank_zero_info = silent
        rz.rank_zero_warn = silent
        rz.rank_zero_debug = silent
    except Exception:
        pass

    try:
        import pytorch_lightning.utilities.rank_zero as rz_old
        rz_old.rank_zero_info = silent
        rz_old.rank_zero_warn = silent
        rz_old.rank_zero_debug = silent
    except Exception:
        pass

    # 🔥 HARD KILL: disable logging module spam
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)

_kill_lightning_logs()

import argparse
import csv
import gc
import json
import os
import random
import warnings
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import torch
from tqdm.auto import tqdm
import logging

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    audio_path:    str
    bho_ref:       str             = ""
    hin_ref:       str             = ""
    start_sec:     Optional[float] = None
    end_sec:       Optional[float] = None
    asr_raw:       str             = ""
    asr_corrected: str             = ""
    nllb_draft:    str             = ""    # FINAL MT output
    stage2_candidates: List[str] = field(default_factory=list)
    stage3_candidates: List[str] = field(default_factory=list)

@dataclass
class Scores:
    wer_raw:        float = 0.0
    cer_raw:        float = 0.0
    bleu_raw:       float = 0.0
    chrf_raw:       float = 0.0
    wer_corrected:  float = 0.0
    cer_corrected:  float = 0.0
    bleu_corrected: float = 0.0
    chrf_corrected: float = 0.0
    bleu_nllb:      float = 0.0
    chrf_nllb:      float = 0.0
    # COMET (reference-based, wmt22-comet-da)
    comet_corrected: float = 0.0   # Stage 2: src=asr_raw,       hyp=asr_corrected, ref=bho_ref
    comet_nllb:      float = 0.0   # Stage 3: src=asr_corrected, hyp=nllb_draft,    ref=hin_ref
    n_samples:      int   = 0
    n_with_bho_ref: int   = 0
    n_with_hin_ref: int   = 0

    def to_dict(self) -> dict:
        return {
            "n_samples":            self.n_samples,
            "n_with_bho_ref":       self.n_with_bho_ref,
            "n_with_hin_ref":       self.n_with_hin_ref,
            "stage1_asr_raw":       {"wer": round(self.wer_raw, 2),       "cer": round(self.cer_raw, 2),       "bleu": round(self.bleu_raw, 2),       "chrf": round(self.chrf_raw, 2)},
            "stage2_asr_corrected": {"wer": round(self.wer_corrected, 2), "cer": round(self.cer_corrected, 2), "bleu": round(self.bleu_corrected, 2), "chrf": round(self.chrf_corrected, 2), "comet": round(self.comet_corrected, 4)},
            "stage3_nllb_final":    {"bleu": round(self.bleu_nllb, 2),    "chrf": round(self.chrf_nllb, 2),    "comet": round(self.comet_nllb, 4)},
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Data loader
# ══════════════════════════════════════════════════════════════════════════════

class IWSLTDataLoader:
    AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

    def __init__(self, data_root: str, split: str = "dev",
                 num_samples: Optional[int] = None):
        self.data_root   = Path(data_root)
        self.split       = split
        self.num_samples = num_samples

    def load(self) -> List[Sample]:
        stamped  = self.data_root / "stamped.tsv"
        segments = self.data_root / "segments.tsv"
        if stamped.exists():
            samples = self._load_from_tsv(stamped)
        elif segments.exists():
            samples = self._load_from_tsv(segments)
        else:
            samples = self._load_from_dir()
        if not samples:
            raise RuntimeError(f"No samples found in {self.data_root}")
        if self.num_samples:
            samples = samples[: self.num_samples]
        n_bho = sum(1 for s in samples if s.bho_ref)
        n_hin = sum(1 for s in samples if s.hin_ref)
        print(f"[DataLoader] {len(samples)} samples  |  bho_refs={n_bho}  hin_refs={n_hin}")
        return samples

    def _load_from_tsv(self, tsv_path: Path) -> List[Sample]:
        samples: List[Sample] = []
        txt_dir   = self.data_root / "txt"
        hin_file  = self._find_hin_file(txt_dir)
        bho_file  = (txt_dir / f"{self.split}.bho") if txt_dir.exists() else None
        hin_lines = self._read_lines(hin_file)
        bho_lines = self._read_lines(bho_file)
        line_idx  = 0
        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split()
                if parts[0].lower() == "audio_file":
                    continue
                raw_path  = parts[0]
                start_sec = float(parts[1]) if len(parts) > 1 else None
                end_sec   = float(parts[2]) if len(parts) > 2 else None
                ap        = self._resolve_audio(raw_path)
                samples.append(Sample(
                    audio_path=str(ap),
                    bho_ref=bho_lines[line_idx].strip() if line_idx < len(bho_lines) else "",
                    hin_ref=hin_lines[line_idx].strip() if line_idx < len(hin_lines) else "",
                    start_sec=start_sec, end_sec=end_sec,
                ))
                line_idx += 1
        return samples

    def _load_from_dir(self) -> List[Sample]:
        wav_dir   = self.data_root / "wav"
        txt_dir   = self.data_root / "txt"
        bho_file  = (txt_dir / f"{self.split}.bho") if txt_dir.exists() else None
        hin_file  = self._find_hin_file(txt_dir)
        bho_lines = self._read_lines(bho_file)
        hin_lines = self._read_lines(hin_file)
        audio_files = sorted(
            p for p in (wav_dir.iterdir() if wav_dir.exists() else [])
            if p.suffix.lower() in self.AUDIO_EXTS
        )
        if not audio_files:
            raise RuntimeError(f"No audio files in {wav_dir}")
        n = len(audio_files)
        bho_lines += [""] * (n - len(bho_lines))
        hin_lines += [""] * (n - len(hin_lines))
        return [
            Sample(audio_path=str(af), bho_ref=bho_lines[i], hin_ref=hin_lines[i])
            for i, af in enumerate(audio_files)
        ]

    def _find_hin_file(self, txt_dir: Path) -> Optional[Path]:
        if not txt_dir or not txt_dir.exists():
            return None
        for suffix in (f"{self.split}.hin", f"{self.split}.hi"):
            p = txt_dir / suffix
            if p.exists():
                return p
        return None

    @staticmethod
    def _read_lines(path: Optional[Path]) -> List[str]:
        if path is None or not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [l.rstrip("\n") for l in f]

    def _resolve_audio(self, raw_path: str) -> Path:
        import re
        p = Path(raw_path)
        if p.is_absolute() and p.exists():
            return p
        for candidate in [self.data_root / raw_path, self.data_root / "wav" / p.name]:
            if candidate.exists():
                return candidate
    
        # ── Zero-pad month/day in filename: 2026-5-1 → 2026-05-01
        padded_name = re.sub(
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            p.name,
        )
        if padded_name != p.name:
            for candidate in [self.data_root / padded_name,
                              self.data_root / "wav" / padded_name]:
                if candidate.exists():
                    return candidate
    
        return self.data_root / raw_path   # original fallback (will error naturally)


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — Whisper ASR
#
#  Processor and model are both loaded from the local fine-tuned model_dir.
#  No HuggingFace Hub access at all.
# ══════════════════════════════════════════════════════════════════════════════

def _peak_normalize(arr: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.abs(arr).max()
    if peak < 1e-8:
        return arr
    return arr * (target_peak / peak)


class Stage1_ASR:
    SAMPLE_RATE = 16_000
    CHUNK_SAMPLES   = 30 * 16_000   # 480 000 — Whisper's hard context limit
    OVERLAP_SAMPLES =  2 * 16_000   # 32 000  — 2 s overlap to avoid boundary cuts

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        batch_size: int = 4,
        max_audio_sec: int = 30,
        hf_token: Optional[str] = None,
        language: str = "hi",
        task: str = "transcribe",
        temperature: float = 0.2,
    ):
        import librosa
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._librosa      = librosa
        self.batch_size    = batch_size
        self.max_audio_sec = max_audio_sec
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._language = language
        self._task     = task
        self.temperature = temperature
        kw = {"token": hf_token} if hf_token else {}

        print(f"[Stage 1 — Whisper ASR] Loading processor from {model_dir}")
        self.processor = WhisperProcessor.from_pretrained(
            model_dir, local_files_only=True, **kw
        )

        print(f"[Stage 1 — Whisper ASR] Loading model from {model_dir}  →  {self.device}")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.float16,
            local_files_only=True,
            **kw,
        )

        # ── FIX: store forced_decoder_ids for explicit use in generate() only.
        self._forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        self.model.config.forced_decoder_ids = None
        try:
            gc = self.model.generation_config
            if getattr(gc, "forced_decoder_ids",     None) is not None:
                gc.forced_decoder_ids     = None
            if getattr(gc, "suppress_tokens",        None) is not None:
                gc.suppress_tokens        = None
            if getattr(gc, "begin_suppress_tokens",  None) is not None:
                gc.begin_suppress_tokens  = None
        except Exception:
            pass

        self.model.eval().to(self.device)
        print(f"[Stage 1 — Whisper ASR] Ready  (batch={batch_size}, lang={language}, task={task})")

    def run(self, samples: List[Sample]) -> List[Sample]:
        for s in tqdm(samples, desc="[Stage 1] Whisper ASR", unit="sample"):
            arr    = self._load_audio(s)
            chunks = self._chunk_array(arr)
            parts  = []
            for i in range(0, len(chunks), self.batch_size):
                parts.extend(self._forward(chunks[i : i + self.batch_size]))
            s.asr_raw = " ".join(p.strip() for p in parts if p.strip())
        return samples

    def _load_audio(self, s: Sample) -> np.ndarray:
        arr, _ = self._librosa.load(s.audio_path, sr=self.SAMPLE_RATE, mono=True)
        if s.start_sec is not None:
            start = int(s.start_sec * self.SAMPLE_RATE)
            end   = int(s.end_sec   * self.SAMPLE_RATE) if s.end_sec else len(arr)
            arr   = arr[start:end]
        return _peak_normalize(arr.astype(np.float32))

    def _chunk_array(self, arr: np.ndarray) -> List[np.ndarray]:
        """Split into ≤30 s chunks with 2 s overlap. Never drops short tail chunks."""
        if len(arr) <= self.CHUNK_SAMPLES:
            return [arr]
        chunks, start = [], 0
        while start < len(arr):
            chunks.append(arr[start : start + self.CHUNK_SAMPLES])
            start += self.CHUNK_SAMPLES - self.OVERLAP_SAMPLES
        return chunks

    def _forward(self, arrays: List[np.ndarray]) -> List[str]:
        inputs = self.processor(
            arrays,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device, dtype=torch.float16)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language=self._language,
                task=self._task,
                no_repeat_ngram_size=3,
                temperature=self.temperature
            )

        return self.processor.batch_decode(outputs, skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
#  QE Fusion — Algorithm 1 (Vernikos et al., 2024)
#  "Don't Rank, Combine! Combining Machine Translation Hypotheses Using
#   Quality Estimation"  https://github.com/GeorgeVern/qe-fusion
# ══════════════════════════════════════════════════════════════════════════════

def _find_diffs(hbase: str, candidates: List[str]) -> Dict[str, List[str]]:
    """
    Find word-level divergent spans between hbase and all other candidates.

    Uses difflib.SequenceMatcher (edit-distance based) as described in the
    paper.  Returns a dict mapping each divergent base_span → [alt_span, ...].

    Insertions (base_span = "") and deletions (alt_span = "") are included;
    pure substitutions produce non-empty spans on both sides.
    """
    base_words = hbase.split()
    diffs: Dict[str, Set[str]] = {}

    for cand in candidates:
        if cand == hbase:
            continue
        cand_words = cand.split()
        sm = SequenceMatcher(None, base_words, cand_words, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            base_span = " ".join(base_words[i1:i2])
            alt_span  = " ".join(cand_words[j1:j2])
            if base_span == alt_span:
                continue
            diffs.setdefault(base_span, set()).add(alt_span)

    return {k: list(v) for k, v in diffs.items()}


def _qe_fusion(
    candidates: List[str],
    source: str,
    qe_scorer: "QEFusionScorer",
    beam_size: int = 4,
) -> str:
    """
    Algorithm 1: QE Fusion (Vernikos et al., 2024).

    Parameters
    ----------
    candidates : N hypotheses generated for a single source sentence.
    source     : Source sentence fed to the QE metric.
    qe_scorer  : QEFusionScorer instance providing .score(sources, hyps).
    beam_size  : Beam width b (paper default: 4 or 5).

    Returns
    -------
    The highest-scoring fused hypothesis.
    """
    candidates = [c for c in candidates if c.strip()]
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # ── Line 1: select top-ranked candidate as base hypothesis
    init_scores = qe_scorer.score([source] * len(candidates), candidates)
    hbase       = candidates[int(np.argmax(init_scores))]

    # ── Line 2: initialise beam
    hyps: List[str] = [hbase]

    # ── Line 3: find divergent spans between hbase and all other candidates
    diffs = _find_diffs(hbase, candidates)

    # ── Lines 4-16: iterative span substitution + beam pruning
    for base_span, alter_spans in diffs.items():
        # Lines 5-11: expand beam with span substitutions
        expanded: List[str] = list(hyps)
        for h in hyps:
            for span in alter_spans:
                # Replace only the first (leftmost) occurrence to stay
                # faithful to the positional semantics of the diff.
                hnew = h.replace(base_span, span, 1) if base_span else h + (" " + span if span else "")
                if hnew not in expanded:
                    expanded.append(hnew)

        if len(expanded) == len(hyps):
            # No new hypotheses were added for this span; skip scoring.
            continue

        # Lines 13-15: score all, sort, keep top-b
        new_scores  = qe_scorer.score([source] * len(expanded), expanded)
        sorted_pair = sorted(
            zip(expanded, new_scores), key=lambda x: x[1], reverse=True
        )
        hyps = [h for h, _ in sorted_pair[:beam_size]]

    # ── Line 17 (Output): return highest-scoring hypothesis
    return hyps[0]


class QEFusionScorer:
    """
    Thin wrapper around COMET-Kiwi (reference-free quality estimation).

    Default model: Unbabel/wmt22-cometkiwi-da
    Scores are in [0, 1] (higher = better translation quality).

    Install dependency:
        pip install unbabel-comet

    Stage 2 usage (ASR correction, Bhojpuri → Bhojpuri):
        source     = noisy Bhojpuri ASR text
        hypothesis = LLM-corrected Bhojpuri text
        COMET-Kiwi was trained on MT pairs; applying it to monolingual
        ASR correction is an approximation that still provides a useful
        relative ranking signal between candidates.

    Stage 3 usage (MT, Bhojpuri → Hindi):
        source     = clean Bhojpuri text
        hypothesis = NLLB Hindi translation
    """

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-cometkiwi-da",
        gpus: int = -1,       # -1 = auto-detect (1 if CUDA, else 0)
        batch_size: int = 32,
    ):
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as exc:
            raise ImportError(
                "unbabel-comet is required for QE Fusion.\n"
                "Install with:  pip install unbabel-comet"
            ) from exc

        self.batch_size = batch_size
        self._gpus      = (1 if torch.cuda.is_available() else 0) if gpus == -1 else gpus

        print(f"\n[QE Scorer] Loading {model_name} ...")
        ckpt        = download_model(model_name)
        self.model  = load_from_checkpoint(ckpt)
        print(f"[QE Scorer] Ready  (gpus={self._gpus}, batch={batch_size})")

    def score(self, sources: List[str], hypotheses: List[str]) -> List[float]:
        """Return a QE score per (src, hyp) pair.  Higher = better."""
        data   = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
        output = self.model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self._gpus,
            progress_bar=False,
        )
        return list(output.scores)


# ══════════════════════════════════════════════════════════════════════════════
#  COMET scorer  (reference-based evaluation)
#
#  Distinct from QEFusionScorer:
#    QEFusionScorer  — reference-FREE  (wmt22-cometkiwi-da)  used at decode time
#    COMETScorer     — reference-BASED (wmt22-comet-da)      used at eval time
#
#  Scores are in [0, 1] (higher = better).
#
#  Stage 2 usage:
#    source    = noisy ASR text (asr_raw)
#    hypothesis= LLM-corrected Bhojpuri (asr_corrected)
#    reference = ground-truth Bhojpuri  (bho_ref)
#
#  Stage 3 usage:
#    source    = clean Bhojpuri (asr_corrected)
#    hypothesis= NLLB Hindi     (nllb_draft)
#    reference = ground-truth Hindi (hin_ref)
#
#  Install dependency:
#      pip install unbabel-comet
# ══════════════════════════════════════════════════════════════════════════════

class COMETScorer:
    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        gpus: int = -1,       # -1 = auto-detect
        batch_size: int = 32,
    ):
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as exc:
            raise ImportError(
                "unbabel-comet is required for COMET scoring.\n"
                "Install with:  pip install unbabel-comet"
            ) from exc

        self.batch_size = batch_size
        self._gpus      = (1 if torch.cuda.is_available() else 0) if gpus == -1 else gpus

        print(f"\n[COMET Scorer] Loading {model_name} ...")
        ckpt       = download_model(model_name)
        self.model = load_from_checkpoint(ckpt)
        print(f"[COMET Scorer] Ready  (gpus={self._gpus}, batch={batch_size})")

    def corpus_score(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> float:
        """
        Compute corpus-level COMET score.

        Filters out triplets where either the reference or hypothesis is empty,
        so partial reference coverage is handled gracefully.

        Returns the mean sentence score (scalar in [0, 1]) or 0.0 if no valid
        triplets exist.
        """
        valid = [
            (s, h, r)
            for s, h, r in zip(sources, hypotheses, references)
            if r.strip() and h.strip()
        ]
        if not valid:
            return 0.0
        srcs, hyps, refs = map(list, zip(*valid))
        data   = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        output = self.model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self._gpus,
            progress_bar=False,
        )
        # output.system_score is the corpus-level mean; fall back to mean of
        # sentence scores for older comet versions that don't expose it.
        if hasattr(output, "system_score"):
            return float(output.system_score)
        return float(np.mean(output.scores))


# ══════════════════════════════════════════════════════════════════════════════
#  LLM loaders  (Stage 2)
# ══════════════════════════════════════════════════════════════════════════════

def _bnb_config():
    from transformers import BitsAndBytesConfig
    try:
        import bitsandbytes  # noqa
        print("[LLM] 4-bit NF4 quantisation active")
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except ImportError:
        print("[LLM] bitsandbytes not found — loading in bf16")
        return None


def _silence_llm_config(model) -> None:
    """Null max_length from both config objects to prevent generate() warnings."""
    try:
        if getattr(model.config, "max_length", None) is not None:
            model.config.max_length = None
    except Exception:
        pass
    try:
        gc_obj = getattr(model, "generation_config", None)
        if gc_obj is not None:
            if getattr(gc_obj, "max_length", None) is not None:
                gc_obj.max_length = None
            if getattr(gc_obj, "temperature", None) is not None:
                gc_obj.temperature = None
            if getattr(gc_obj, "top_p", None) is not None:
                gc_obj.top_p = None
    except Exception:
        pass


def load_finetuned_llm(checkpoint_path: str, hf_token: Optional[str] = None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"token": hf_token} if hf_token else {}
    cfg_path = Path(checkpoint_path) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {checkpoint_path}.\n"
            "Have you run training yet?  If not, use --zeroshot instead.")
    with open(cfg_path) as f:
        base_name = json.load(f).get("base_model_name_or_path", checkpoint_path)
    print(f"[LLM — fine-tuned] Base: {base_name}  Adapter: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True, **kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb      = _bnb_config()
    dtype_kw = {} if bnb is not None else {"dtype": torch.bfloat16}
    try:
        print("[LLM — fine-tuned] Loading base model from local cache ...")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, local_files_only=True, **dtype_kw, **kw,
        )
    except Exception:
        print(f"[LLM — fine-tuned] Not in local cache — downloading {base_name} ...")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, local_files_only=False, **dtype_kw, **kw,
        )

    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    _silence_llm_config(model)
    print("[LLM — fine-tuned] Ready")
    return model, tokenizer


def load_zeroshot_llm(base_model_name: str, hf_token: Optional[str] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"token": hf_token} if hf_token else {}
    print(f"[LLM — zero-shot] Loading: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, **kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb      = _bnb_config()
    dtype_kw = {} if bnb is not None else {"dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, **dtype_kw, **kw,
    )
    model.eval()
    _silence_llm_config(model)
    print("[LLM — zero-shot] Ready")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — prompt  (verbatim copy from train_asr_corrector.py)
# ══════════════════════════════════════════════════════════════════════════════

ASR_CORRECT_INSTRUCTION = (
    "नीचे दिए गए भोजपुरी ASR आउटपुट को सुधारें। "
    "केवल सुधरा हुआ भोजपुरी वाक्य लिखें — कोई अतिरिक्त व्याख्या नहीं।\n\n"
    "Correct the following Bhojpuri ASR output. "
    "Keep the meaning EXACTLY the same — do not add or invent any new content which change the semantic of the sentence."
    "Write ONLY the corrected Bhojpuri sentence — no explanation, no commentary.\n\n"
    "Noisy ASR: {noisy}"
)


def _build_stage2_prompt(noisy: str, bos: str = "<s>") -> str:
    """
    Mirrors build_prompt() in train_asr_corrector.py:

        <s><|user|>
        {instruction}
        <|assistant|>

    BOS is prepended here; call tokenizer with add_special_tokens=False
    to prevent a duplicate BOS insertion.
    """
    instruction = ASR_CORRECT_INSTRUCTION.format(noisy=noisy.strip())
    return f"{bos}<|user|>\n{instruction}\n<|assistant|>\n"


# ══════════════════════════════════════════════════════════════════════════════
#  Output post-processor  (Stage 2)
# ══════════════════════════════════════════════════════════════════════════════

_EXPLANATION_TRIGGERS = (
    "इसका अनुवाद",    "अनुवाद होगा",   "का अर्थ",
    "हिंदी में",        "यह वाक्य",      "यह अनुवाद",
    "नोट:",             "टिप्पणी:",      "ध्यान दें",
    "corrected bhojpuri:", "corrected:",
)


def _extract_first_devanagari_line(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    for q in ('"', "'", '\u201c', '\u201d', '\u2018', '\u2019'):
        text = text.strip(q)
    text = text.strip()
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        if sum(1 for c in line if c.isascii() and c.isalpha()) > 8:
            continue
        line_lower = line.lower()
        if any(line_lower.startswith(t.lower()) or t.lower() in line_lower[:40]
               for t in _EXPLANATION_TRIGGERS):
            continue
        return line
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  Shared LLM generation  (Stage 2)
#
#  QE Fusion mode (when qe_scorer is not None):
#    • Generates n_candidates per prompt via nucleus sampling (p=0.9, T=0.6).
#    • Groups candidates back to their source prompt (1:n_candidates mapping).
#    • Calls _qe_fusion() per sentence to select/recombine the best hypothesis.
#
#  Greedy mode (qe_scorer is None — original behaviour):
#    • do_sample=False, repetition_penalty=1.1  (unchanged from v1).
#
#  ALIGNMENT GUARANTEE
#  -------------------
#  results is built by iterating batches in order and appending one entry per
#  input prompt.  The final list is always len(prompts) long.
# ══════════════════════════════════════════════════════════════════════════════

def _llm_generate(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    desc: str = "LLM generate",
    post_processor: Callable[[str], str] = _extract_first_devanagari_line,
    # ── QE Fusion (optional) ─────────────────────────────────────────────────
    qe_scorer: Optional[QEFusionScorer] = None,
    n_candidates: int = 5,
    source_texts: Optional[List[str]] = None,
    qe_beam_size: int = 4,
    return_candidates: bool = False,
) -> List[str]:
    """
    Generate one output string per prompt.

    When qe_scorer is provided:
      - Generates n_candidates per prompt via nucleus sampling (p=0.9, T=0.6).
      - Applies _qe_fusion() to select the best hypothesis per sentence.
      - source_texts[i] is passed as the QE metric source for prompt i.
        If source_texts is None, an empty string is used as a fallback
        (scores will still provide a useful relative ranking).

    When qe_scorer is None:
      - Greedy decode with repetition_penalty=1.1 (original behaviour).
    """
    device   = next(model.parameters()).device
    use_qe   = qe_scorer is not None
    results: List[str] = []
    all_candidates: List[List[str]] = []

    if use_qe:
        print(f"[{desc}] QE Fusion active  "
              f"(n_candidates={n_candidates}, beam_size={qe_beam_size})")

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
        chunk     = prompts[i : i + batch_size]
        # Source texts used by QE scorer (noisy ASR for Stage 2).
        src_chunk = (
            source_texts[i : i + batch_size]
            if source_texts is not None
            else [""] * len(chunk)
        )

        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
            add_special_tokens=False,   # BOS already in prompt string
        ).to(device)

        if use_qe:
            # ── QE Fusion path: nucleus sampling, n_candidates per prompt
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                num_return_sequences=n_candidates,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            # ── Original greedy-decode path (unchanged)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        # input_len is uniform within a batch because of left-padding.
        input_len = enc["input_ids"].shape[1]

        if use_qe:
            # out.shape = (len(chunk) * n_candidates, seq_len)
            # Slice group j → out[j*n_cand : (j+1)*n_cand]
            for j, src in enumerate(src_chunk):
                group      = out[j * n_candidates : (j + 1) * n_candidates]
                cand_texts = [
                    post_processor(
                        tokenizer.decode(
                            ids[input_len:], skip_special_tokens=True
                        ).strip()
                    )
                    for ids in group
                ]
                cand_texts = [c for c in cand_texts if c.strip()]
                if not cand_texts:
                    all_candidates.append([])
                    results.append("")
                else:
                    all_candidates.append(cand_texts)
                    results.append(
                        _qe_fusion(cand_texts, src, qe_scorer, beam_size=qe_beam_size)
                    )
        else:
            # out.shape = (len(chunk), seq_len)
            for ids in out:
                raw = tokenizer.decode(ids[input_len:], skip_special_tokens=True).strip()
                results.append(post_processor(raw))

    # Sanity-check: output list must be 1-to-1 with input prompts.
    assert len(results) == len(prompts), (
        f"[_llm_generate] alignment error: {len(results)} results for "
        f"{len(prompts)} prompts — this is a bug, please report it."
    )
    if return_candidates:
        return results, all_candidates
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — LLM ASR corrector
# ══════════════════════════════════════════════════════════════════════════════

class Stage2_ASRCorrector:
    def __init__(
        self,
        mode: str,
        checkpoint_path: Optional[str] = None,
        base_model_name: Optional[str] = None,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        hf_token: Optional[str] = None,
        # ── QE Fusion (optional) ─────────────────────────────────────────────
        qe_scorer: Optional[QEFusionScorer] = None,
        qe_n_candidates: int = 5,
        qe_beam_size: int = 4,
    ):
        self.mode            = mode
        self.batch_size      = batch_size
        self.max_new_tokens  = max_new_tokens
        self.qe_scorer       = qe_scorer
        self.qe_n_candidates = qe_n_candidates
        self.qe_beam_size    = qe_beam_size
        self.model           = None
        self.tokenizer       = None

        qe_tag = "  [+QE Fusion]" if qe_scorer is not None else ""

        if mode == "finetuned":
            print(f"\n[Stage 2] Mode: fine-tuned QLoRA{qe_tag}")
            self.model, self.tokenizer = load_finetuned_llm(checkpoint_path, hf_token)
        elif mode == "zeroshot":
            print(f"\n[Stage 2] Mode: zero-shot (base LLM, no adapter){qe_tag}")
            self.model, self.tokenizer = load_zeroshot_llm(base_model_name, hf_token)
        elif mode == "skip":
            print("\n[Stage 2] Mode: SKIP — raw ASR passed to Stage 3")
        else:
            raise ValueError(f"Unknown Stage2 mode: {mode!r}")

    def run(self, samples: List[Sample]) -> List[Sample]:
        if self.mode == "skip":
            for s in samples:
                s.asr_corrected = s.asr_raw
            return samples

        bos     = self.tokenizer.bos_token or "<s>"
        prompts = [_build_stage2_prompt(s.asr_raw, bos=bos) for s in samples]

        # Source texts for QE scorer = noisy ASR inputs.
        # Passed only when QE Fusion is active; ignored otherwise.
        source_texts = [s.asr_raw for s in samples] if self.qe_scorer else None

        if self.qe_scorer:
            preds, cand_bank = _llm_generate(
                self.model, self.tokenizer, prompts,
                self.max_new_tokens, self.batch_size,
                "[Stage 2] ASR correction",
                post_processor=_extract_first_devanagari_line,
                qe_scorer=self.qe_scorer,
                n_candidates=self.qe_n_candidates,
                source_texts=source_texts,
                qe_beam_size=self.qe_beam_size,
                return_candidates=True,
            )
        else:
            preds = _llm_generate(
                self.model, self.tokenizer, prompts,
                self.max_new_tokens, self.batch_size,
                "[Stage 2] ASR correction",
                post_processor=_extract_first_devanagari_line,
            )
            cand_bank = [[] for _ in samples]

        n_fallback = 0
        for s, pred, cands in zip(samples, preds, cand_bank):
            s.stage2_candidates = cands
            if pred.strip():
                s.asr_corrected = pred.strip()
            else:
                s.asr_corrected = s.asr_raw
                n_fallback += 1

        if n_fallback:
            print(f"[Stage 2] ⚠  {n_fallback}/{len(samples)} empty outputs → fell back to asr_raw")

        return samples


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — NLLB  (final MT stage)
#
#  QE Fusion mode (when qe_scorer is not None):
#    Replaces beam-search with epsilon sampling (ε=0.02, T=0.5) to generate
#    n_candidates per sentence, then selects/recombines via _qe_fusion().
#    Source for QE scorer = clean Bhojpuri text (asr_corrected).
#
#  Beam-search mode (qe_scorer is None — original behaviour):
#    num_beams=self.num_beams  (unchanged).
# ══════════════════════════════════════════════════════════════════════════════

class Stage3_NLLB:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_src_length: int = 256,
        max_new_tokens: int = 192,
        num_beams: int = 4,
        src_lang: str = "bho_Deva",
        tgt_lang: str = "hin_Deva",
        # ── QE Fusion (optional) ─────────────────────────────────────────────
        qe_scorer: Optional[QEFusionScorer] = None,
        qe_n_candidates: int = 5,
        qe_beam_size: int = 4,
        qe_epsilon: float = 0.02,
    ):
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, NllbTokenizer
        self.batch_size      = batch_size
        self.max_src_length  = max_src_length
        self.max_new_tokens  = max_new_tokens
        self.num_beams       = num_beams
        self.src_lang        = src_lang
        self.tgt_lang        = tgt_lang
        self.device          = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.qe_scorer       = qe_scorer
        self.qe_n_candidates = qe_n_candidates
        self.qe_beam_size    = qe_beam_size
        self.qe_epsilon = qe_epsilon

        qe_tag = "  [+QE Fusion]" if qe_scorer is not None else ""
        print(f"\n[Stage 3 — NLLB] Loading {model_path}  →  {self.device}{qe_tag}")

        self.tokenizer = NllbTokenizer.from_pretrained(
            model_path, src_lang=src_lang, tgt_lang=tgt_lang)
        _cfg = AutoConfig.from_pretrained(model_path)
        _cfg.tie_word_embeddings = False
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=_cfg)
        self.model.config.max_length = None
        try:
            gc_obj = getattr(self.model, "generation_config", None)
            if gc_obj is not None and getattr(gc_obj, "max_length", None) is not None:
                gc_obj.max_length = None
        except Exception:
            pass
        self.model.eval().to(self.device)

        if qe_scorer is not None:
            print(
                f"[Stage 3 — NLLB] QE Fusion  "
                f"(epsilon_cutoff=0.02, T=0.5, n_candidates={qe_n_candidates}, "
                f"qe_beam={qe_beam_size})"
            )
        else:
            print(
                f"[Stage 3 — NLLB] Beam search  "
                f"(batch={batch_size}, beams={num_beams}, "
                f"src_max={max_src_length}, tgt_max={max_new_tokens})"
            )

    def run(self, samples: List[Sample]) -> List[Sample]:
        texts = [s.asr_corrected for s in samples]
        for i in tqdm(range(0, len(texts), self.batch_size),
                      desc="[Stage 3] NLLB MT", unit="batch"):
            hyps, cand_bank = self._forward(texts[i : i + self.batch_size])
            for s, h, cands in zip(samples[i : i + self.batch_size], hyps, cand_bank):
                s.stage3_candidates = cands
                s.nllb_draft = h
        return samples

    def _forward(self, texts: List[str]) -> List[str]:
        self.tokenizer.src_lang = self.src_lang
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_src_length, return_tensors="pt",
        ).to(self.device)
        tgt_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        if self.qe_scorer is not None:
            # ── QE Fusion path: epsilon sampling for diverse candidates
            # Epsilon sampling (Hewitt et al., 2022): tokens with probability
            # below ε are masked.  Combined with temperature T=0.5 this gives
            # diverse yet high-quality candidates (paper §4).
            n = self.qe_n_candidates
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    forced_bos_token_id=tgt_id,
                    do_sample=True,
                    epsilon_cutoff=self.qe_epsilon,    # ε from Hewitt et al. (2022)
                    temperature=0.5,
                    num_return_sequences=n,
                    max_new_tokens=self.max_new_tokens,
                )
            # out.shape = (len(texts) * n, seq_len)
            all_hyps = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            results = []
            cand_bank = []
            
            for i, src in enumerate(texts):
                cands = all_hyps[i * n : (i + 1) * n]
                cands = [c for c in cands if c.strip()]
                cand_bank.append(cands)
                if not cands:
                    results.append("")
                else:
                    results.append(_qe_fusion(cands, src, self.qe_scorer, beam_size=self.qe_beam_size))
            
            return results, cand_bank

        else:
            # ── Original beam-search path (unchanged)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    forced_bos_token_id=tgt_id,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                )
            return self.tokenizer.batch_decode(out, skip_special_tokens=True), [[] for _ in texts]


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _safe_asr_metrics(hyps, refs):
    import sacrebleu
    from jiwer import cer as jiwer_cer, wer as jiwer_wer
    valid = [(h, r) for h, r in zip(hyps, refs) if r.strip()]
    if not valid:
        return {"wer": 0.0, "cer": 0.0, "bleu": 0.0, "chrf": 0.0}
    h, r = map(list, zip(*valid))
    return {
        "wer":  round(jiwer_wer(r, h) * 100, 2),
        "cer":  round(jiwer_cer(r, h) * 100, 2),
        "bleu": round(sacrebleu.corpus_bleu(h, [r]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2),
    }


def _safe_mt_metrics(hyps, refs):
    import sacrebleu
    valid = [(h, r) for h, r in zip(hyps, refs) if r.strip()]
    if not valid:
        return {"bleu": 0.0, "chrf": 0.0}
    h, r = map(list, zip(*valid))
    return {
        "bleu": round(sacrebleu.corpus_bleu(h, [r]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(h, [r], word_order=2).score, 2),
    }


def evaluate(samples: List[Sample],
             asr_calc_metrics: bool,
             mt_calc_metrics: bool,
             comet_scorer: Optional[COMETScorer] = None) -> Scores:
    sc = Scores(
        n_samples=len(samples),
        n_with_bho_ref=sum(1 for s in samples if s.bho_ref.strip()),
        n_with_hin_ref=sum(1 for s in samples if s.hin_ref.strip()),
    )

    if asr_calc_metrics and sc.n_with_bho_ref > 0:
        print("\n[Metrics] Computing ASR metrics ...")
        bho_refs = [s.bho_ref for s in samples]
        raw_m = _safe_asr_metrics([s.asr_raw       for s in samples], bho_refs)
        cor_m = _safe_asr_metrics([s.asr_corrected for s in samples], bho_refs)
        sc.wer_raw,       sc.cer_raw,       sc.bleu_raw,       sc.chrf_raw       = raw_m["wer"], raw_m["cer"], raw_m["bleu"], raw_m["chrf"]
        sc.wer_corrected, sc.cer_corrected, sc.bleu_corrected, sc.chrf_corrected = cor_m["wer"], cor_m["cer"], cor_m["bleu"], cor_m["chrf"]
        print(f"  S1 raw     WER={sc.wer_raw:.2f}%  CER={sc.cer_raw:.2f}%  BLEU={sc.bleu_raw:.2f}  chrF++={sc.chrf_raw:.2f}")
        print(f"  S2 corr    WER={sc.wer_corrected:.2f}%  CER={sc.cer_corrected:.2f}%  BLEU={sc.bleu_corrected:.2f}  chrF++={sc.chrf_corrected:.2f}")

        if comet_scorer is not None:
            print("[Metrics] Computing COMET for Stage 2 ...")
            sc.comet_corrected = comet_scorer.corpus_score(
                sources    =[s.asr_raw       for s in samples],
                hypotheses =[s.asr_corrected for s in samples],
                references =[s.bho_ref       for s in samples],
            )
            print(f"  S2 corr    COMET={sc.comet_corrected:.4f}")
    elif asr_calc_metrics:
        print(
            "[Metrics] ⚠  ASR metrics SKIPPED — no Bhojpuri references found.\n"
            "          Check that txt/{split}.bho exists under your --data_root."
        )

    if mt_calc_metrics and sc.n_with_hin_ref > 0:
        print("\n[Metrics] Computing MT metrics ...")
        hin_refs = [s.hin_ref for s in samples]
        nllb_m   = _safe_mt_metrics([s.nllb_draft for s in samples], hin_refs)
        sc.bleu_nllb, sc.chrf_nllb = nllb_m["bleu"], nllb_m["chrf"]
        print(f"  S3 NLLB    BLEU={sc.bleu_nllb:.2f}  chrF++={sc.chrf_nllb:.2f}")

        if comet_scorer is not None:
            print("[Metrics] Computing COMET for Stage 3 ...")
            sc.comet_nllb = comet_scorer.corpus_score(
                sources    =[s.asr_corrected for s in samples],
                hypotheses =[s.nllb_draft    for s in samples],
                references =[s.hin_ref       for s in samples],
            )
            print(f"  S3 NLLB    COMET={sc.comet_nllb:.4f}")
    elif mt_calc_metrics:
        print(
            "[Metrics] ⚠  MT metrics SKIPPED — no Hindi references found.\n"
            "          Check that txt/dev.hin or txt/dev.hi exists under your --data_root."
        )

    return sc


# ══════════════════════════════════════════════════════════════════════════════
#  Sample printer
# ══════════════════════════════════════════════════════════════════════════════

def print_samples(samples: List[Sample], n: int = 3) -> None:
    print(f"\n{'═'*72}\n  PIPELINE OUTPUT SAMPLES  (n={min(n, len(samples))})\n{'═'*72}")
    for i, idx in enumerate(random.sample(range(len(samples)), min(n, len(samples))), 1):
        s = samples[idx]
        print(f"\n  [{i}] AUDIO       : {Path(s.audio_path).name}")
        print(f"       S1 ASR raw  : {s.asr_raw}")
        print(f"       S2 corrected: {s.asr_corrected}")
        if s.bho_ref:
            print(f"       BHO ref     : {s.bho_ref}")
        print(f"       S3 NLLB     : {s.nllb_draft}  ← final")
        if s.hin_ref:
            print(f"       HIN ref     : {s.hin_ref}")
    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Output writer
# ══════════════════════════════════════════════════════════════════════════════

class OutputWriter:
    def __init__(self, output_dir: str, split: str):
        self.output_dir = Path(output_dir)
        self.txt_dir    = self.output_dir / "txt"
        self.split      = split

    def write(self, samples: List[Sample], scores: Scores) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.txt_dir.mkdir(parents=True, exist_ok=True)

        for fname, lines in {
            f"{self.split}.bho.raw":  [s.asr_raw       for s in samples],
            f"{self.split}.bho.hyp":  [s.asr_corrected for s in samples],
            f"{self.split}.hin.hyp":  [s.nllb_draft    for s in samples],
        }.items():
            p = self.txt_dir / fname
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  {fname:<28} → {p}")

        self._write_stamped_tsv(samples)
        self._write_train_hi(samples)
        self._write_qe_candidates(samples)

        path = self.output_dir / "scores.json"
        path.write_text(json.dumps(scores.to_dict(), indent=2), encoding="utf-8")
        print(f"  {'scores.json':<28} → {path}")

        self._write_summary(scores)
        print(f"\n[OutputWriter] All outputs saved to {self.output_dir}/")
    
    def _write_qe_candidates(self, samples: List[Sample]) -> None:
        path = self.output_dir / "qe_candidates.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                rec = {
                    "audio": Path(s.audio_path).name,
                    "stage2": {
                        "source": s.asr_raw,
                        "selected": s.asr_corrected,
                        "candidates": s.stage2_candidates,
                    },
                    "stage3": {
                        "source": s.asr_corrected,
                        "selected": s.nllb_draft,
                        "candidates": s.stage3_candidates,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {'qe_candidates.jsonl':<28} → {path}")
            

    def _write_stamped_tsv(self, samples: List[Sample]) -> None:
        path = self.output_dir / "stamped.tsv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            for s in samples:
                w.writerow([
                    Path(s.audio_path).name,
                    s.asr_raw, s.asr_corrected,
                    s.nllb_draft,
                    s.bho_ref, s.hin_ref,
                ])
        print(f"  {'stamped.tsv':<28} → {path}")

    def _write_train_hi(self, samples: List[Sample]) -> None:
        path = self.output_dir / "train.hi"
        path.write_text("\n".join(s.nllb_draft for s in samples) + "\n", encoding="utf-8")
        print(f"  {'train.hi':<28} → {path}")

    def _write_summary(self, scores: Scores) -> None:
        path = self.output_dir / "scores_summary.txt"

        def _comet_tag(val: float, has_ref: bool) -> str:
            if not has_ref:
                return "N/A (no ref)"
            return f"{val:.4f}" if val > 0.0 else "N/A (COMET disabled)"

        lines = [
            "=" * 60,
            "  Experiment 1 — Bhojpuri → Hindi  3-Stage Cascade",
            "  (Stage 4 LLM post-editor removed; NLLB is final output)",
            "=" * 60,
            f"  Samples total  : {scores.n_samples}",
            f"  w/ bho_ref     : {scores.n_with_bho_ref}",
            f"  w/ hin_ref     : {scores.n_with_hin_ref}",
            "",
            "  ASR METRICS",
            (
                f"  S1 raw     WER={scores.wer_raw:.2f}%  CER={scores.cer_raw:.2f}%  BLEU={scores.bleu_raw:.2f}  chrF++={scores.chrf_raw:.2f}"
                if scores.n_with_bho_ref > 0 else
                "  S1 raw     N/A (no bho_ref)"
            ),
            (
                f"  S2 corr    WER={scores.wer_corrected:.2f}%  CER={scores.cer_corrected:.2f}%  BLEU={scores.bleu_corrected:.2f}  chrF++={scores.chrf_corrected:.2f}  COMET={_comet_tag(scores.comet_corrected, scores.n_with_bho_ref > 0)}"
                if scores.n_with_bho_ref > 0 else
                "  S2 corr    N/A (no bho_ref)"
            ),
            "",
            "  MT METRICS  (NLLB final)",
            f"  S3 NLLB    BLEU={scores.bleu_nllb:.2f}  chrF++={scores.chrf_nllb:.2f}  COMET={_comet_tag(scores.comet_nllb, scores.n_with_hin_ref > 0)}",
            "=" * 60,
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {'scores_summary.txt':<28} → {path}")
        print("\n" + "\n".join(lines))


def _cleanup(label: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if label:
        print(f"[Memory] GPU cache cleared after {label}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 1 — 3-stage Bhojpuri→Hindi cascade inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--asr_model",  required=True,
                   help="Path to fine-tuned Whisper model directory.")
    p.add_argument("--mt_model",   required=True)
    p.add_argument("--data_root",  required=True)
    p.add_argument("--asr_corrector_model", default=None)

    whisper_grp = p.add_argument_group("Whisper ASR (Stage 1)")
    whisper_grp.add_argument("--whisper_language", default="hi")
    whisper_grp.add_argument("--whisper_task",     default="transcribe",
                             choices=["transcribe", "translate"])

    zs = p.add_argument_group("Zero-shot mode (Stage 2)")
    zs.add_argument("--zeroshot",            action="store_true")
    zs.add_argument("--zeroshot_base_model", default=None)
    zs.add_argument("--zeroshot_mode",       default="full", choices=["full", "skip_s2"])

    # ── QE Fusion ──────────────────────────────────────────────────────────
    qe = p.add_argument_group("QE Fusion (Stages 2 and/or 3)")
    qe.add_argument(
        "--qe_fusion",
        default="both",
        choices=["both", "llm", "mt", "none"],
        help=(
            "Apply QE Fusion decoding.  "
            "'both' (default): Stage 2 (LLM) + Stage 3 (NLLB).  "
            "'llm': Stage 2 only.  "
            "'mt': Stage 3 only.  "
            "'none': disable; use original greedy/beam-search."
        ),
    )
    qe.add_argument(
        "--qe_model",
        default="Unbabel/wmt22-cometkiwi-da",
        help="HuggingFace model name for COMET-Kiwi QE scorer.",
    )
    qe.add_argument(
        "--qe_n_candidates",
        type=int,
        default=5,
        help="Number of candidates to generate per sentence for QE Fusion.",
    )
    qe.add_argument(
        "--qe_beam_size",
        type=int,
        default=4,
        help="Beam width b in Algorithm 1 (top-b hypotheses retained per span).",
    )
    qe.add_argument(
        "--qe_scorer_batch",
        type=int,
        default=32,
        help="Batch size for the COMET-Kiwi QE scorer.",
    )
    qe.add_argument(
    "--qe_epsilon",
    type=float,
    default=0.02,
    help="Epsilon cutoff for MT QE Fusion sampling."
    )
    # ── COMET evaluation ───────────────────────────────────────────────────
    comet = p.add_argument_group("COMET evaluation (reference-based)")
    comet.add_argument(
        "--comet_model",
        default="Unbabel/wmt22-comet-da",
        help="HuggingFace model name for reference-based COMET scoring.",
    )
    comet.add_argument(
        "--comet_scorer_batch",
        type=int,
        default=32,
        help="Batch size for the COMET evaluation scorer.",
    )
    comet.add_argument(
        "--skip_comet",
        action="store_true",
        help=(
            "Disable COMET evaluation entirely.  "
            "Useful when unbabel-comet is not installed or to save time."
        ),
    )
    # ───────────────────────────────────────────────────────────────────────
    p.add_argument("--split",       default="dev")
    p.add_argument("--output_dir",  default="./exp1_outputs")
    p.add_argument("--num_samples", type=int, default=None)

    p.add_argument("--asr_calc_metrics", action="store_true")
    p.add_argument("--mt_calc_metrics",  action="store_true")

    p.add_argument("--asr_batch",   type=int, default=4)
    p.add_argument("--asr_max_sec", type=int, default=30)

    p.add_argument("--corrector_batch",      type=int, default=4)
    p.add_argument("--corrector_max_tokens", type=int, default=100)

    p.add_argument("--nllb_batch",      type=int, default=16)
    p.add_argument("--nllb_beams",      type=int, default=4)
    p.add_argument("--nllb_src_maxlen", type=int, default=256)
    p.add_argument("--nllb_tgt_maxlen", type=int, default=256)
    p.add_argument("--src_lang", default="bho_Deva")
    p.add_argument("--tgt_lang", default="hin_Deva")
    p.add_argument("--temp_asr", type=float, default=0.2)

    p.add_argument("--hf_token",          default=None)
    p.add_argument("--num_print_samples", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()

    if args.zeroshot:
        if args.zeroshot_mode == "full" and not args.zeroshot_base_model:
            raise ValueError(
                "--zeroshot_base_model is required when "
                "--zeroshot and --zeroshot_mode=full"
            )
    else:
        if not args.asr_corrector_model:
            raise ValueError(
                "--asr_corrector_model is required unless --zeroshot is set.\n"
                "To skip Stage 2 entirely, use: --zeroshot --zeroshot_mode=skip_s2"
            )

    # ── Determine which stages get QE Fusion ─────────────────────────────
    use_qe_llm = args.qe_fusion in ("both", "llm")
    use_qe_mt  = args.qe_fusion in ("both", "mt")

    # Stage 2 is not run when zeroshot_mode=skip_s2, so suppress warning.
    if use_qe_llm and args.zeroshot and args.zeroshot_mode == "skip_s2":
        print("[QE Fusion] ⚠  --qe_fusion includes 'llm' but Stage 2 is skipped "
              "(--zeroshot_mode=skip_s2).  QE Fusion will only be applied to Stage 3.")
        use_qe_llm = False

    mode_label = (
        f"ZERO-SHOT  mode={args.zeroshot_mode}  base={args.zeroshot_base_model}"
        if args.zeroshot else "FINE-TUNED"
    )
    if args.qe_fusion == "none":
        qe_label = "QE Fusion=DISABLED"
    else:
        _qe_stages = {
            "both": "LLM + MT",
            "llm":  "LLM only",
            "mt":   "MT only",
        }[args.qe_fusion]
        _eps_tag = f"  epsilon={args.qe_epsilon}" if args.qe_fusion in ("both", "mt") else ""
        qe_label = (
            f"QE Fusion={_qe_stages}  model={args.qe_model}  "
            f"n_cand={args.qe_n_candidates}  beam={args.qe_beam_size}{_eps_tag}"
        )

    print(
        f"\n{'═'*60}\n"
        f"  Experiment 1 — 3-Stage Bhojpuri → Hindi Pipeline\n"
        f"{'═'*60}"
    )
    print(f"  mode           : {mode_label}")
    print(f"  split          : {args.split}")
    print(f"  data_root      : {args.data_root}")
    print(f"  asr_model      : {args.asr_model}  (processor + weights)")
    print(f"  whisper_lang   : {args.whisper_language}  task={args.whisper_task}")
    print(f"  nllb_src_maxlen: {args.nllb_src_maxlen}")
    print(f"  nllb_tgt_maxlen: {args.nllb_tgt_maxlen}")
    print(f"  {qe_label}")
    if args.num_samples:
        print(f"  num_samples    : {args.num_samples}")
    print(f"{'═'*60}\n")

    samples = IWSLTDataLoader(args.data_root, args.split, args.num_samples).load()

    # ── Load QE scorer once and share across stages ───────────────────────
    qe_scorer: Optional[QEFusionScorer] = None
    if args.qe_fusion != "none":
        qe_scorer = QEFusionScorer(
            model_name=args.qe_model,
            batch_size=args.qe_scorer_batch,
        )

    # ── Load COMET scorer (reference-based evaluation) ────────────────────
    # Loaded once up-front so the weights are in memory when evaluate() runs.
    # Skipped if --skip_comet is set, if neither metric flag is active, or if
    # references will not be available (detected lazily inside evaluate()).
    comet_scorer: Optional[COMETScorer] = None
    if not args.skip_comet and (args.asr_calc_metrics or args.mt_calc_metrics):
        try:
            comet_scorer = COMETScorer(
                model_name=args.comet_model,
                batch_size=args.comet_scorer_batch,
            )
        except ImportError as e:
            print(f"[COMET] ⚠  {e}\n[COMET] Skipping COMET scoring.")

    # ── Stage 1: Whisper ASR ──────────────────────────────────────────────
    stage1 = Stage1_ASR(
        model_dir=args.asr_model,
        batch_size=args.asr_batch,
        max_audio_sec=args.asr_max_sec,
        hf_token=args.hf_token,
        language=args.whisper_language,
        task=args.whisper_task,
    )
    samples = stage1.run(samples)
    del stage1
    _cleanup("Stage 1")

    # ── Stage 2: LLM ASR corrector ────────────────────────────────────────
    s2_mode = (
        "skip"      if args.zeroshot and args.zeroshot_mode == "skip_s2"
        else "zeroshot" if args.zeroshot
        else "finetuned"
    )
    stage2 = Stage2_ASRCorrector(
        s2_mode,
        checkpoint_path=args.asr_corrector_model,
        base_model_name=args.zeroshot_base_model,
        batch_size=args.corrector_batch,
        max_new_tokens=args.corrector_max_tokens,
        hf_token=args.hf_token,
        # QE Fusion
        qe_scorer=qe_scorer if use_qe_llm else None,
        qe_n_candidates=args.qe_n_candidates,
        qe_beam_size=args.qe_beam_size,
    )
    samples = stage2.run(samples)
    del stage2
    _cleanup("Stage 2")

    # ── Stage 3: NLLB MT ──────────────────────────────────────────────────
    stage3 = Stage3_NLLB(
        args.mt_model,
        batch_size=args.nllb_batch,
        max_src_length=args.nllb_src_maxlen,
        max_new_tokens=args.nllb_tgt_maxlen,
        num_beams=args.nllb_beams,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        # QE Fusion
        qe_scorer=qe_scorer if use_qe_mt else None,
        qe_n_candidates=args.qe_n_candidates,
        qe_beam_size=args.qe_beam_size,
        qe_epsilon=args.qe_epsilon,
    )
    samples = stage3.run(samples)
    del stage3
    _cleanup("Stage 3")

    scores = evaluate(samples, args.asr_calc_metrics, args.mt_calc_metrics,
                      comet_scorer=comet_scorer)
    print_samples(samples, n=args.num_print_samples)
    OutputWriter(args.output_dir, args.split).write(samples, scores)

    print(f"\n[Done] {len(samples)} samples processed.")


if __name__ == "__main__":
    main()