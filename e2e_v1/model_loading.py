"""
model_loading.py
----------------
Loads:
  • Wav2Vec2 encoder  — from a Hugging Face repo/URL, a filesender.cesnet.cz
                         tarball URL, or a local path
  • NLLB decoder      — from a filesender.cesnet.cz tarball URL, or a local path

Both tarballs are expected to contain a HuggingFace saved_pretrained directory
(i.e. config.json + pytorch_model.bin / model.safetensors) once extracted.

Internal tarball structure assumed (either layout works):
    model.tar.gz
    └── <any_top_dir>/          ← single top-level folder (auto-detected)
            config.json
            pytorch_model.bin   (or model.safetensors)
            tokenizer files     (for NLLB)
            ...

Builds:
  • LayerAggregator   — learnable weighted sum over configurable encoder layers
  • LengthAdapter     — Conv1d stride downsampler + projection to NLLB hidden dim
  • InterConnectionST — full encoder–adapter–decoder model

Usage
-----
    from model_loading import build_model, build_processor_and_tokenizer

    processor, tokenizer = build_processor_and_tokenizer(
        wav2vec2_url = "Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60",
        nllb_url     = "https://filesender.cesnet.cz/download.php?token=YOUR_TOKEN_B&files_ids=YOUR_FILE_ID",
        cache_dir    = "./model_cache",
    )
    model = build_model(
        wav2vec2_url       = "...",
        nllb_url           = "https://filesender.cesnet.cz/download.php?token=...&files_ids=...",
        cache_dir          = "./model_cache",
        aggregation_layers = [6, 12, 18, 24],
    )
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from transformers.modeling_outputs import BaseModelOutput

SRC_LANG = "bho_Deva"
TGT_LANG = "hin_Deva"

# Fallback hub IDs — used only for config/tokenizer if tarball lacks them
WAV2VEC2_HUB = "Harveenchadha/vakyansh-wav2vec2-bhojpuri-bhom-60"
NLLB_HUB     = "facebook/nllb-200-distilled-600M"


# ---------------------------------------------------------------------------
# Tarball downloader + extractor
# ---------------------------------------------------------------------------

def _token_from_url(url: str) -> str:
    """Extract the token= value from a filesender URL for use as a cache key."""
    try:
        qs = parse_qs(urlparse(url).query)
        return qs.get("token", [url[-16:]])[0]
    except Exception:
        return url[-16:]


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _is_filesender_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host == "filesender.cesnet.cz"


def _validate_filesender_url(url: str) -> None:
    """
    Require a direct FileSender file URL instead of the browser download page.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    if not qs.get("token", [None])[0]:
        raise ValueError(
            "FileSender URL is missing a token=... query parameter."
        )

    if parsed.path in {"/download.php", "download.php"}:
        if not qs.get("files_ids", [None])[0]:
            raise ValueError(
                "Direct FileSender download URLs must include files_ids=...."
            )
        return

    if qs.get("s", [None])[0] == "download":
        raise ValueError(
            "The provided FileSender URL points to the browser download page, "
            "not the direct file URL. Use the direct link instead, for example "
            "'https://filesender.cesnet.cz/download.php?token=...&files_ids=...'."
        )


def _read_file_prefix(path: Path, num_bytes: int = 2048) -> bytes:
    with open(path, "rb") as f:
        return f.read(num_bytes)


def _raise_invalid_archive(
    archive_path: Path,
    source: str,
    *,
    response_url: Optional[str] = None,
    content_type: Optional[str] = None,
) -> None:
    sample = _read_file_prefix(archive_path)
    sample_text = sample.decode("utf-8", errors="ignore").strip().lower()

    reason = "Downloaded file is not a valid .tar or .tar.gz archive."
    if "html" in (content_type or "").lower() or "<html" in sample_text or "<!doctype" in sample_text:
        reason = "Downloaded HTML instead of a model archive."
    elif sample_text.startswith("{") or sample_text.startswith("["):
        reason = "Downloaded JSON/text instead of a model archive."

    details = [reason, f"Source: {source}"]
    if response_url and response_url != source:
        details.append(f"Final URL: {response_url}")
    if content_type:
        details.append(f"Content-Type: {content_type}")

    if _is_filesender_url(source):
        details.append(
            "Use the direct FileSender file URL "
            "('download.php?token=...&files_ids=...') or a local tarball path."
        )
        details.append(
            "If you opened a '?s=download&token=...' page in the browser, copy "
            "the actual file download request instead of the page URL."
        )

    raise ValueError(" ".join(details))


def _safe_extractall(tar: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in tar.getmembers():
        member_path = (root / member.name).resolve()
        if os.path.commonpath([str(root), str(member_path)]) != str(root):
            raise ValueError(
                f"Refusing to extract '{member.name}' outside {destination}"
            )
    tar.extractall(path=destination)


def _extract_tarball(
    archive_path: Path,
    target_dir: Path,
    *,
    source: str,
    content_type: Optional[str] = None,
    response_url: Optional[str] = None,
) -> Path:
    if not tarfile.is_tarfile(archive_path):
        _raise_invalid_archive(
            archive_path,
            source,
            response_url=response_url,
            content_type=content_type,
        )

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f"{target_dir.name}_extract_", dir=str(target_dir.parent))
    )
    try:
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                _safe_extractall(tar, staging_dir)
        except tarfile.ReadError:
            with tarfile.open(archive_path, "r:") as tar:
                _safe_extractall(tar, staging_dir)

        _find_model_dir(staging_dir)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(staging_dir), str(target_dir))
        return _find_model_dir(target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _normalize_hf_repo_id(source: str) -> Optional[str]:
    """
    Return a Hugging Face repo ID if `source` points to the Hub.

    Supports:
      - repo IDs like "namespace/model"
      - URLs like "https://huggingface.co/namespace/model"
      - URLs with deeper paths such as "/tree/main" or "/resolve/main/..."
    """
    source = source.strip()

    if source.startswith("hf://"):
        repo_id = source[5:].strip("/")
        return repo_id or None

    if _is_url(source):
        parsed = urlparse(source)
        host = parsed.netloc.lower()
        if host not in {"huggingface.co", "www.huggingface.co"}:
            return None

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(
                f"Could not parse Hugging Face repo from URL: {source}"
            )
        if parts[0] in {"datasets", "spaces"}:
            raise ValueError(
                "Expected a model repo on huggingface.co, got a non-model URL: "
                f"{source}"
            )
        return "/".join(parts[:2])

    # Existing local paths should stay local. Non-existent "namespace/model"
    # strings are treated as Hub repo IDs.
    if Path(source).exists():
        return None

    cleaned = source.strip("/").replace("\\", "/")
    if cleaned.count("/") == 1 and not cleaned.endswith((".tar", ".tar.gz")):
        return cleaned

    return None


def download_and_extract(
    url: str,
    cache_dir: str,
    name: str,
    force: bool = False,
) -> str:
    """
    Download a tarball from `url`, extract it under `cache_dir/{name}/`,
    and return the path to the extracted model directory.

    Handles both .tar.gz and .tar formats. For FileSender, this expects the
    direct file URL (for example download.php?...&files_ids=...), not the
    browser download page URL.

    Args:
        url       : full filesender.cesnet.cz download URL
        cache_dir : local directory to cache downloads and extractions
        name      : logical name ("wav2vec2" or "nllb") used for subfolder
        force     : re-download even if already cached

    Returns:
        path to the extracted directory containing config.json etc.
    """
    cache_dir  = Path(cache_dir)
    target_dir = cache_dir / name
    done_flag  = target_dir / ".extracted"

    if done_flag.exists() and not force:
        model_dir = _find_model_dir(target_dir)
        print(f"[model_loading] Using cached {name} at: {model_dir}")
        return str(model_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = cache_dir / f"{name}.tar.gz"

    # ---- download ----
    print(f"[model_loading] Downloading {name} from filesender...")
    print(f"  URL: {url}")

    if _is_filesender_url(url):
        _validate_filesender_url(url)

    # filesender redirects to actual file; urllib follows redirects automatically
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        }
    )
    with urllib.request.urlopen(req, timeout=3600) as response, \
         open(tarball_path, "wb") as out_f:
        response_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 1024    # 1 MB
        while True:
            buf = response.read(chunk)
            if not buf:
                break
            out_f.write(buf)
            downloaded += len(buf)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB  ({pct:.1f}%)", end="", flush=True)
    print()
    print(f"[model_loading] Download complete: {tarball_path}")

    # ---- extract ----
    print(f"[model_loading] Extracting {name}...")
    try:
        model_dir = _extract_tarball(
            tarball_path,
            target_dir,
            source=url,
            content_type=content_type,
            response_url=response_url,
        )
        done_flag.touch()
    finally:
        tarball_path.unlink(missing_ok=True)   # free disk space
    print(f"[model_loading] Extracted {name} → {model_dir}")
    return str(model_dir)


def _find_model_dir(root: Path) -> Path:
    """
    Find the directory inside `root` that contains config.json.
    Handles both flat layout (root/config.json) and nested
    (root/<some_folder>/config.json).
    """
    # Direct
    if (root / "config.json").exists():
        return root
    # One level deep
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "config.json").exists():
            return child
    # Two levels deep (some tarballs nest twice)
    for child in sorted(root.iterdir()):
        if child.is_dir():
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and (grandchild / "config.json").exists():
                    return grandchild
    raise FileNotFoundError(
        f"Could not find config.json anywhere under {root}. "
        f"Contents: {list(root.rglob('config.json'))}"
    )


def _has_any_file(root: Path, filenames: List[str]) -> bool:
    return any((root / filename).exists() for filename in filenames)


def _resolve_nllb_tokenizer_source(nllb_path: str) -> str:
    """
    Prefer tokenizer assets from the extracted NLLB directory when present.
    Fall back to the canonical NLLB Hub tokenizer for checkpoint-only exports.
    """
    path = Path(nllb_path)
    if not path.exists() or not path.is_dir():
        return nllb_path

    tokenizer_files = [
        "sentencepiece.bpe.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    if _has_any_file(path, tokenizer_files):
        return nllb_path

    print(
        "[model_loading] NLLB checkpoint does not include tokenizer files; "
        f"falling back to Hugging Face tokenizer: {NLLB_HUB}"
    )
    return NLLB_HUB


def _resolve_model_path(
    url_or_path: str,
    cache_dir: str,
    name: str,
) -> str:
    """
    If `url_or_path` looks like a URL, download + extract and return local path.
    Otherwise treat as a local directory path.
    """
    if _is_url(url_or_path):
        return download_and_extract(url_or_path, cache_dir, name)
    # Local path
    p = Path(url_or_path)
    # If it's a tarball file, extract it
    if p.is_file() and (url_or_path.endswith(".tar.gz") or url_or_path.endswith(".tar")):
        target_dir = Path(cache_dir) / name
        target_dir.mkdir(parents=True, exist_ok=True)
        done_flag  = target_dir / ".extracted"
        if not done_flag.exists():
            print(f"[model_loading] Extracting local tarball {p}...")
            _extract_tarball(p, target_dir, source=str(p))
            done_flag.touch()
        return str(_find_model_dir(target_dir))
    # Assume it's already an extracted directory
    assert p.exists(), f"Model path not found: {p}"
    return str(p)


def _resolve_wav2vec2_source(
    wav2vec2_source: str,
    cache_dir: str,
) -> str:
    """
    Resolve Wav2Vec2 to either:
      - a Hugging Face repo ID, or
      - a local extracted directory/tarball path
    """
    repo_id = _normalize_hf_repo_id(wav2vec2_source)
    if repo_id is not None:
        print(f"[model_loading] Using Wav2Vec2 from Hugging Face Hub: {repo_id}")
        return repo_id

    if _is_url(wav2vec2_source) and not _is_filesender_url(wav2vec2_source):
        raise ValueError(
            "Wav2Vec2 URL must be either a Hugging Face model URL or a "
            f"filesender.cesnet.cz download URL, got: {wav2vec2_source}"
        )

    return _resolve_model_path(wav2vec2_source, cache_dir, "wav2vec2")


# ---------------------------------------------------------------------------
# Processor & Tokenizer
# ---------------------------------------------------------------------------

def build_processor_and_tokenizer(
    wav2vec2_url: str,
    nllb_url: str,
    cache_dir: str = "./model_cache",
):
    """
    Returns (Wav2Vec2Processor, NllbTokenizer).

    Args:
        wav2vec2_url : Hugging Face repo/URL, filesender URL, or local path/tarball
        nllb_url     : filesender URL *or* local path/tarball for NLLB
        cache_dir    : where to cache downloads
    """
    w2v_source = _resolve_wav2vec2_source(wav2vec2_url, cache_dir)
    nllb_path = _resolve_model_path(nllb_url,     cache_dir, "nllb")
    tokenizer_source = _resolve_nllb_tokenizer_source(nllb_path)

    processor = Wav2Vec2Processor.from_pretrained(
        w2v_source,
        cache_dir=str(Path(cache_dir) / "huggingface"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )
    print(f"[model_loading] Processor loaded from: {w2v_source}")
    print(f"[model_loading] Tokenizer loaded from:  {tokenizer_source}")
    return processor, tokenizer


# ---------------------------------------------------------------------------
# Layer Aggregator
# ---------------------------------------------------------------------------

class LayerAggregator(nn.Module):
    """
    Learnable weighted sum of selected wav2vec2 hidden layers.
    Weights are softmax-normalised scalars — always positive, sum to 1.

    Args:
        aggregation_layers : 0-based layer indices to aggregate.
                             [] = all layers (CNN feat + all transformer).
        num_encoder_layers : total transformer layers in the encoder.
    """

    def __init__(self, aggregation_layers: List[int], num_encoder_layers: int):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers

        if len(aggregation_layers) == 0:
            self.layer_indices = list(range(num_encoder_layers + 1))
        else:
            for idx in aggregation_layers:
                assert 0 <= idx <= num_encoder_layers, (
                    f"Layer index {idx} out of range [0, {num_encoder_layers}]"
                )
            self.layer_indices = sorted(aggregation_layers)

        # one learnable logit per selected layer; uniform after softmax at init
        self.layer_weights = nn.Parameter(torch.zeros(len(self.layer_indices)))

    def forward(self, hidden_states: tuple) -> torch.Tensor:
        """
        Args:
            hidden_states: tuple of (B, T, D) tensors, one per layer.
                           Index 0 = CNN features, 1..N = transformer layers.
        Returns:
            (B, T, D) weighted sum
        """
        weights  = F.softmax(self.layer_weights, dim=0)           # (n,)
        selected = torch.stack(
            [hidden_states[i] for i in self.layer_indices], dim=0
        )                                                          # (n, B, T, D)
        return (weights[:, None, None, None] * selected).sum(dim=0)


# ---------------------------------------------------------------------------
# Length Adapter
# ---------------------------------------------------------------------------

class LengthAdapter(nn.Module):
    """
    Reduces acoustic sequence length via strided Conv1d, then projects
    to NLLB hidden dim.

    2 × stride-2 convs → 4× reduction (default, good starting point).
    Try stride=4 or num_conv_layers=3 if sequences are still too long.
    """

    def __init__(
        self,
        enc_dim: int,
        dec_dim: int,
        stride: int = 2,
        num_conv_layers: int = 2,
    ):
        super().__init__()
        layers = []
        for _ in range(num_conv_layers):
            kernel  = 2 * stride - 1
            padding = (kernel - 1) // 2
            layers += [
                nn.Conv1d(enc_dim, enc_dim, kernel_size=kernel,
                          stride=stride, padding=padding),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(enc_dim)
        self.proj = nn.Linear(enc_dim, dec_dim)

    def forward(
        self,
        hidden: torch.Tensor,                       # (B, T, D_enc)
        attention_mask: Optional[torch.Tensor],     # (B, T)
    ):
        x = hidden.transpose(1, 2)                  # (B, D_enc, T)
        x = self.conv(x)                            # (B, D_enc, T')
        x = x.transpose(1, 2)                       # (B, T', D_enc)
        x = self.norm(x)
        x = self.proj(x)                            # (B, T', D_dec)

        adapted_mask = None
        if attention_mask is not None:
            mask_f = attention_mask.float().unsqueeze(1)   # (B, 1, T)
            for layer in self.conv:
                if isinstance(layer, nn.Conv1d):
                    mask_f = F.avg_pool1d(
                        mask_f,
                        kernel_size=layer.stride[0],
                        stride=layer.stride[0],
                        padding=0,
                        ceil_mode=True,
                    )
            adapted_mask = (mask_f.squeeze(1) > 0).long()

        return x, adapted_mask


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class InterConnectionST(nn.Module):
    """
    Inter-connection Speech Translation:
        Wav2Vec2 (frozen) → LayerAggregator → LengthAdapter → NLLB decoder
    """

    def __init__(
        self,
        wav2vec2_model: Wav2Vec2Model,
        nllb_model,
        aggregation_layers: List[int],
        adapter_stride: int = 2,
        adapter_num_convs: int = 2,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        self.encoder = wav2vec2_model
        self.nllb    = nllb_model

        num_enc_layers = self.encoder.config.num_hidden_layers
        enc_dim        = self.encoder.config.hidden_size
        dec_dim        = self.nllb.config.d_model

        self.aggregator    = LayerAggregator(aggregation_layers, num_enc_layers)
        self.length_adapter = LengthAdapter(enc_dim, dec_dim, adapter_stride, adapter_num_convs)

        if freeze_encoder:
            self._freeze_encoder()
        if freeze_decoder:
            self._freeze_decoder()

        self._print_param_summary()

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("[model_loading] Wav2Vec2 encoder frozen.")

    def _freeze_decoder(self):
        self.set_decoder_trainable(False)
        print("[model_loading] NLLB decoder frozen.")

    def unfreeze_encoder_top_layers(self, num_layers: int = 4):
        total = self.encoder.config.num_hidden_layers
        for i, layer in enumerate(self.encoder.encoder.layers):
            if i >= total - num_layers:
                for p in layer.parameters():
                    p.requires_grad = True
        print(f"[model_loading] Unfroze top {num_layers} encoder layers.")

    def set_decoder_trainable(self, trainable: bool = True):
        for p in self.nllb.parameters():
            p.requires_grad = trainable

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[model_loading] Params — total: {total/1e6:.1f}M | "
            f"trainable: {trainable/1e6:.1f}M | "
            f"frozen: {(total-trainable)/1e6:.1f}M"
        )

    def _build_encoder_attention_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Convert raw-sample attention masks to wav2vec2 feature-frame masks.
        """
        if attention_mask is None:
            return None

        return self.encoder._get_feature_vector_attention_mask(
            hidden.shape[1],
            attention_mask,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ):
        enc_out    = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        aggregated = self.aggregator(enc_out.hidden_states)
        enc_mask   = self._build_encoder_attention_mask(aggregated, attention_mask)
        adapted, ada_mask = self.length_adapter(aggregated, enc_mask)

        return self.nllb(
            encoder_outputs=BaseModelOutput(last_hidden_state=adapted),
            attention_mask=ada_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        forced_bos_token_id: int,
        max_new_tokens: int = 200,
        num_beams: int = 4,
        **kwargs,
    ):
        enc_out    = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        aggregated = self.aggregator(enc_out.hidden_states)
        enc_mask   = self._build_encoder_attention_mask(aggregated, attention_mask)
        adapted, ada_mask = self.length_adapter(aggregated, enc_mask)

        return self.nllb.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=adapted),
            attention_mask=ada_mask,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_model(
    wav2vec2_url: str,
    nllb_url: str,
    aggregation_layers: List[int],
    cache_dir: str = "./model_cache",
    adapter_stride: int = 2,
    adapter_num_convs: int = 2,
    freeze_encoder: bool = True,
    freeze_decoder: bool = False,
) -> InterConnectionST:
    """
    Downloads (if needed), extracts, and loads both models, then builds
    the full InterConnectionST.

    Args:
        wav2vec2_url       : Hugging Face repo/URL, filesender URL, or local path/tarball
        nllb_url           : filesender URL or local path/tarball
        aggregation_layers : wav2vec2 layer indices to aggregate.
                             Examples:
                               [6, 12, 18, 24]   every 6th (large)
                               []                all layers
        cache_dir          : local cache for downloads
        adapter_stride     : conv stride in LengthAdapter
        adapter_num_convs  : number of strided conv layers
        freeze_encoder     : freeze wav2vec2 weights (recommended)
        freeze_decoder     : freeze NLLB decoder weights to save GPU memory
    """
    w2v_source = _resolve_wav2vec2_source(wav2vec2_url, cache_dir)
    nllb_path = _resolve_model_path(nllb_url,     cache_dir, "nllb")

    print(f"[model_loading] Loading Wav2Vec2 from: {w2v_source}")
    wav2vec2 = Wav2Vec2Model.from_pretrained(
        w2v_source,
        cache_dir=str(Path(cache_dir) / "huggingface"),
    )

    print(f"[model_loading] Loading NLLB from: {nllb_path}")
    nllb = AutoModelForSeq2SeqLM.from_pretrained(nllb_path)

    model = InterConnectionST(
        wav2vec2_model=wav2vec2,
        nllb_model=nllb,
        aggregation_layers=aggregation_layers,
        adapter_stride=adapter_stride,
        adapter_num_convs=adapter_num_convs,
        freeze_encoder=freeze_encoder,
        freeze_decoder=freeze_decoder,
    )
    return model
