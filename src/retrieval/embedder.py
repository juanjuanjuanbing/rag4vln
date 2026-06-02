# -*- coding: utf-8 -*-
"""
Retrieval embeddings: ``Embedder`` and implementations.

- **BERTEmbedder**: ``transformers`` BERT-style model, masked mean pooling + L2 (``text_emb`` only).
- **SentenceBERTEmbedder** / **BGEEmbedder**: ``sentence_transformers`` sentence vectors (SBERT, BGE, etc.).
- **build_text_embedder_from_config**: pick one of the above via ``retrieval.text_embedder`` or explicit ``backend`` in ``src/config.yaml``.
- **ViTEmbedder**: fills ``image_feat`` only (KB view images).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

from ..config_io import DEFAULT_CONFIG_PATH, load_retrieval_config


def default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def _load_embedder_config(path: Path) -> Dict[str, Any]:
    return load_retrieval_config(path)


class Embedder(ABC):
    """Embedder API: text → ``text_emb``, image → ``image_feat``; dimension must match ``score_all``."""

    version: str = "embedder-stub-v0"

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output embedding dimension D (text and vision must match in retrieval)."""
        raise NotImplementedError

    @abstractmethod
    def embed(
        self,
        instruction: str,
        image: Any = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        *,
        image_role: str = "kb_view",
    ) -> Dict[str, Any]:
        """
        - ``text_emb``: written by the text encoder when ``instruction`` is non-empty.
        - ``image_feat``: written by the vision encoder when ``image`` is set (``image_role`` usually ``kb_view``).

        ``position`` / ``rotation`` reserved for future geometry channels; ignored by default.
        """
        raise NotImplementedError


class BinaryRandomEmbedder(Embedder):
    """For testing: deterministic random 0/1 vectors; can simulate both text and image branches."""

    version: str = "binary-random-embedder-v0"

    def __init__(self, dim: int = 768, threshold: float = 0.5):
        self.dim = int(dim)
        self.threshold = float(threshold)
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    @property
    def embedding_dim(self) -> int:
        return self.dim

    def _seed_from(self, instruction: str, image: Any) -> int:
        h = hashlib.md5()
        h.update((instruction or "").encode("utf-8"))
        h.update(b"|")
        h.update(str(id(image)).encode("ascii"))
        return int.from_bytes(h.digest()[:4], "little", signed=False)

    def embed(
        self,
        instruction: str,
        image: Any = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        *,
        image_role: str = "kb_view",
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text_emb": None, "image_feat": None}

        if instruction is not None:
            seed_t = self._seed_from(instruction, image=None)
            if torch is not None:
                g = torch.Generator()
                g.manual_seed(seed_t)
                x = torch.rand(self.dim, generator=g, dtype=torch.float32)
                out["text_emb"] = (x > self.threshold).to(torch.float32)
            else:
                rng = np.random.default_rng(seed_t % (2**32))
                x = rng.random(self.dim, dtype=np.float32)
                out["text_emb"] = (x > self.threshold).astype(np.float32)

        if image is not None:
            seed_i = self._seed_from(instruction or "", image=image)
            if torch is not None:
                g = torch.Generator()
                g.manual_seed(seed_i)
                x = torch.rand(self.dim, generator=g, dtype=torch.float32)
                out["image_feat"] = (x > self.threshold).to(torch.float32)
            else:
                rng = np.random.default_rng(seed_i % (2**32))
                x = rng.random(self.dim, dtype=np.float32)
                out["image_feat"] = (x > self.threshold).astype(np.float32)

        return out


class BERTEmbedder(Embedder):
    """Native BERT (or compatible ``AutoModel``): masked mean pool on ``last_hidden_state`` + L2; ``text_emb`` only."""

    version: str = "bert-mean-pool-v0"

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if torch is None:
            raise ImportError("BERTEmbedder requires torch")
        try:
            from transformers import AutoModel as _AutoModel
            from transformers import AutoTokenizer as _AutoTokenizer
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ImportError("BERTEmbedder requires transformers") from e
        cfg_path = Path(config_path) if config_path is not None else default_config_path()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config not found: {cfg_path}")
        self._cfg: Dict[str, Any] = _load_embedder_config(cfg_path)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_name = (self._cfg.get("bert") or {}).get("pretrained") or "bert-base-uncased"
        self._tokenizer = _AutoTokenizer.from_pretrained(model_name)
        self._model = _AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        self._dim = int(self._model.config.hidden_size)
        expected_d = int(self._cfg.get("embedding_dim", 768))
        if self._dim != expected_d:
            raise ValueError(
                f"embedding_dim in config is {expected_d} but BERT hidden is {self._dim}. "
                "Align embedding_dim with checkpoint hidden_size."
            )

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _encode_text(self, text: str) -> torch.Tensor:
        s = (text or "").strip()
        if not s:
            s = "."
        inputs = self._tokenizer(
            s,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            token_emb = out.last_hidden_state
            attn_mask = inputs["attention_mask"].unsqueeze(-1).to(token_emb.dtype)
            summed = (token_emb * attn_mask).sum(dim=1)
            denom = attn_mask.sum(dim=1).clamp(min=1e-8)
            emb = (summed / denom).squeeze(0)
            emb = torch.nn.functional.normalize(emb, p=2, dim=0)
        return emb.detach().float().cpu()

    def embed(
        self,
        instruction: str,
        image: Any = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        *,
        image_role: str = "kb_view",
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text_emb": None, "image_feat": None}
        if instruction is not None and str(instruction).strip() != "":
            out["text_emb"] = self._encode_text(str(instruction))
        return out


class STBackedTextEmbedder(Embedder):
    """Sentence vectors via ``sentence_transformers.SentenceTransformer`` (SBERT, BGE, etc.)."""

    _cfg_key: str = "sentence_bert"
    _default_model: str = "sentence-transformers/all-mpnet-base-v2"
    version: str = "st-backed-text-v0"

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if torch is None:
            raise ImportError(f"{type(self).__name__} requires torch")
        try:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ImportError(
                f"{type(self).__name__} requires sentence-transformers (pip install sentence-transformers)"
            ) from e
        cfg_path = Path(config_path) if config_path is not None else default_config_path()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config not found: {cfg_path}")
        self._cfg: Dict[str, Any] = _load_embedder_config(cfg_path)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        block = self._cfg.get(self._cfg_key) or {}
        model_name = block.get("pretrained") or self._default_model
        self._model = _SentenceTransformer(model_name, device=self._device)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        expected_d = int(self._cfg.get("embedding_dim", 768))
        if self._dim != expected_d:
            raise ValueError(
                f"embedding_dim in config is {expected_d} but {type(self).__name__} output dim is {self._dim}. "
                "Align embedding_dim with the chosen model."
            )

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _encode_text(self, text: str) -> torch.Tensor:
        s = (text or "").strip()
        if not s:
            s = "."
        emb = self._model.encode(
            s,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(emb, torch.Tensor) and emb.dim() > 1:
            emb = emb.squeeze(0)
        return emb.detach().float().cpu()

    def embed(
        self,
        instruction: str,
        image: Any = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        *,
        image_role: str = "kb_view",
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text_emb": None, "image_feat": None}
        if instruction is not None and str(instruction).strip() != "":
            out["text_emb"] = self._encode_text(str(instruction))
        return out


class SentenceBERTEmbedder(STBackedTextEmbedder):
    """Sentence-BERT: reads ``sentence_bert.pretrained`` from config."""

    _cfg_key = "sentence_bert"
    _default_model = "sentence-transformers/all-mpnet-base-v2"
    version = "sentence-bert-embedder-v1"


class BGEEmbedder(STBackedTextEmbedder):
    """BGE: reads ``bge.pretrained`` from config (same loader as SBERT, different checkpoint)."""

    _cfg_key = "bge"
    _default_model = "BAAI/bge-base-en-v1.5"
    version = "bge-embedder-v0"


_TEXT_BACKEND_ALIASES: Dict[str, str] = {
    "sbert": "sentence_bert",
    "st": "sentence_bert",
    "sentence-bert": "sentence_bert",
}


def build_text_embedder_from_config(
    config_path: Optional[Union[str, Path]] = None,
    *,
    backend: Optional[str] = None,
    device: Optional[str] = None,
) -> Embedder:
    """
    Build a text ``Embedder`` from YAML ``text_embedder`` (or explicit ``backend``).

    ``backend`` / ``text_embedder``: ``bert`` | ``sentence_bert`` | ``bge`` (aliases ``sbert``, ``st`` → sentence_bert).
    """
    cfg_path = Path(config_path) if config_path is not None else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    cfg = _load_embedder_config(cfg_path)
    raw = backend if backend is not None else cfg.get("text_embedder") or "sentence_bert"
    key = str(raw).strip().lower()
    key = _TEXT_BACKEND_ALIASES.get(key, key)
    if key == "bert":
        return BERTEmbedder(config_path=cfg_path, device=device)
    if key == "sentence_bert":
        return SentenceBERTEmbedder(config_path=cfg_path, device=device)
    if key == "bge":
        return BGEEmbedder(config_path=cfg_path, device=device)
    raise ValueError(
        f"Unknown text_embedder {raw!r} (resolved {key!r}); "
        "expected bert, sentence_bert, or bge (aliases: sbert, st)."
    )


class ViTEmbedder(Embedder):
    """ViT: outputs ``image_feat`` from ``image`` only (ignores ``instruction``)."""

    version: str = "vit-embedder-v0"

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if torch is None:
            raise ImportError("ViTEmbedder requires torch")
        try:
            from transformers import AutoImageProcessor as _AutoImageProcessor
            from transformers import ViTModel as _ViTModel
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ImportError("ViTEmbedder requires transformers") from e
        cfg_path = Path(config_path) if config_path is not None else default_config_path()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"config not found: {cfg_path}")
        self._cfg: Dict[str, Any] = _load_embedder_config(cfg_path)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        vit_name = (self._cfg.get("vit") or {}).get("pretrained", "google/vit-base-patch16-224")
        self._vit_processor = _AutoImageProcessor.from_pretrained(vit_name)
        self._vit = _ViTModel.from_pretrained(vit_name).to(self._device)
        self._vit.eval()

        self._dim = int(self._vit.config.hidden_size)
        expected_d = int(self._cfg.get("embedding_dim", 768))
        if self._dim != expected_d:
            raise ValueError(
                f"embedding_dim in config is {expected_d} but ViT hidden is {self._dim}. "
                "Align embedding_dim with checkpoint hidden_size."
            )

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _encode_image(self, image: Any) -> torch.Tensor:
        from .image_utils import pil_from_any

        pil = pil_from_any(image)
        inputs = self._vit_processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._vit(**inputs)
            emb = out.last_hidden_state[:, 0, :].squeeze(0)
        return emb.detach().float().cpu()

    def embed(
        self,
        instruction: str,
        image: Any = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        *,
        image_role: str = "kb_view",
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text_emb": None, "image_feat": None}
        if image is not None:
            out["image_feat"] = self._encode_image(image)
        return out
