# -*- coding: utf-8 -*-
"""
Retrieval framework: `Retriever` + `ScoreCache`; embeddings use **text encoder** + **vision encoder** (or legacy `Embedder`).

Robot observation VLM caption runs inside `retrieve` via `DashScopeRobotCaptioner`, then **text `Embedder`** (e.g. `BERTEmbedder` / `SentenceBERTEmbedder` / `BGEEmbedder`).
"""

from __future__ import annotations

import hashlib
import heapq
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

import numpy as np

from ..config_io import DEFAULT_CONFIG_PATH, load_retrieval_config
from ..kb import KnowledgeBase

from .caption import DashScopeRobotCaptioner, build_robot_captioner
from .embedder import BGEEmbedder, Embedder


class ScoreCache:
    """Score cache (minimal in-memory interface)."""

    def __init__(self, max_entries: int = 200_000):
        self._max = max_entries
        self._data: Dict[Any, Any] = {}

    def get(self, key: Any) -> Optional[Any]:
        return self._data.get(key)

    def set(self, key: Any, value: Any) -> None:
        # Stub: no eviction policy yet; LRU can be added later.
        if len(self._data) > self._max:
            self._data.clear()
        self._data[key] = value

    def clear(self) -> None:
        self._data.clear()


class Retriever:
    """
    Minimal retriever skeleton.

    - `score_all`: batch-compute all scores at once (torch)
    - `retrieve`: pass **`text_embedder`** + **`vision_embedder`** (`Embedder` subclasses); robot image caption uses the VLM in `retrieve`, then **text embedder** with `instruction=caption`.
    """

    def __init__(
        self,
        text_embedder: Optional[Embedder] = None,
        vision_embedder: Optional[Embedder] = None,
        score_cache: Optional[ScoreCache] = None,
        enable_score_cache: bool = True,
        robot_captioner: Optional[DashScopeRobotCaptioner] = None,
        caption_config_path: Optional[Union[str, Path]] = None,
    ):
        self._text_embedder = text_embedder
        self._vision_embedder = vision_embedder
        self._score_cache = score_cache if score_cache is not None else ScoreCache()
        self._enable_score_cache = enable_score_cache

        if text_embedder is None or vision_embedder is None:
            raise ValueError("text_embedder and vision_embedder are required")
        if text_embedder.embedding_dim != vision_embedder.embedding_dim:
            raise ValueError(
                f"text_embedder.embedding_dim={text_embedder.embedding_dim} != "
                f"vision_embedder.embedding_dim={vision_embedder.embedding_dim}"
            )
        self._robot_captioner = robot_captioner
        if self._robot_captioner is None:
            cap_path = Path(caption_config_path) if caption_config_path is not None else None
            self._robot_captioner = build_robot_captioner(cap_path)

        # Retrieval mix coeffs from unified config retrieval section; default 0
        self._alpha = 0.0
        self._beta = 0.0
        self._score_norm_mode = "minmax"
        self._score_norm_eps = 1e-6
        # Default False when omitted in YAML (legacy no-normalization behavior)
        self._normalize_visual_semantic_scores = False
        # Ablation: start/end view match skips scene_belonging_score (zones still use scene score)
        self._no_scene_gate_on_view_scores = False
        self._bge_query_prefix = "Represent this sentence for searching relevant passages: "
        self._kb_embed_cache_mem: Dict[str, Dict[str, Any]] = {}
        cfg_path = Path(caption_config_path) if caption_config_path is not None else DEFAULT_CONFIG_PATH
        try:
            if cfg_path.is_file():
                cfg = load_retrieval_config(cfg_path)
                self._alpha = float(cfg.get("alpha", 0.0))
                self._beta = float(cfg.get("beta", 0.0))
                self._score_norm_mode = str(cfg.get("score_norm_mode", "minmax")).strip().lower()
                self._score_norm_eps = float(cfg.get("score_norm_eps", 1e-6))
                if "normalize_visual_semantic_scores" in cfg:
                    self._normalize_visual_semantic_scores = self._cfg_bool(
                        cfg.get("normalize_visual_semantic_scores"), default=False
                    )
                if "no_scene_gate_on_view_scores" in cfg:
                    self._no_scene_gate_on_view_scores = self._cfg_bool(
                        cfg.get("no_scene_gate_on_view_scores"), default=False
                    )
                if "bge_query_prefix" in cfg:
                    self._bge_query_prefix = str(cfg.get("bge_query_prefix") or "")
        except Exception:
            self._alpha = float(self._alpha)
            self._beta = float(self._beta)
            self._score_norm_mode = str(self._score_norm_mode)
            self._score_norm_eps = float(self._score_norm_eps)
            self._normalize_visual_semantic_scores = bool(self._normalize_visual_semantic_scores)

    @staticmethod
    def _cfg_bool(v: Any, *, default: bool = False) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("0", "false", "no", "off"):
            return False
        if s in ("1", "true", "yes", "on"):
            return True
        return default

    @staticmethod
    def _normalize_scores(x: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
        m = str(mode or "none").strip().lower()
        if m == "none":
            return x
        if m == "zscore":
            mu = x.mean()
            std = x.std(unbiased=False)
            return (x - mu) / (std + eps)
        if m == "minmax":
            mn = x.min()
            mx = x.max()
            return (x - mn) / (mx - mn + eps)
        return x

    def _kb_cache_key(
        self,
        *,
        kb: KnowledgeBase,
        scene_ids: List[str],
        z_cnt_max: int,
        v_cnt_max: int,
        d_dim: int,
        embed_view_images: bool,
    ) -> str:
        h = hashlib.md5()
        h.update(str(getattr(kb, "root", "")).encode("utf-8"))
        h.update(b"|")
        h.update("|".join(scene_ids).encode("utf-8"))
        h.update(b"|")
        h.update(f"{z_cnt_max}|{v_cnt_max}|{d_dim}|{int(embed_view_images)}".encode("utf-8"))
        h.update(b"|")
        h.update(str(getattr(self._text_embedder, "version", type(self._text_embedder).__name__)).encode("utf-8"))
        h.update(b"|")
        h.update(str(getattr(self._vision_embedder, "version", type(self._vision_embedder).__name__)).encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _load_kb_cache_file(cache_path: Path) -> Optional[Dict[str, Any]]:
        if torch is None:
            return None
        try:
            if not cache_path.is_file():
                return None
            obj = torch.load(cache_path, map_location="cpu")
            if not isinstance(obj, dict):
                return None
            return obj
        except Exception:
            return None

    @staticmethod
    def _save_kb_cache_file(cache_path: Path, payload: Dict[str, Any]) -> None:
        if torch is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_path)
        except Exception:
            return

    @staticmethod
    def _fingerprint_query(
        instruction: str,
        image: Any = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
    ) -> str:
        h = hashlib.md5()
        h.update((instruction or "").encode("utf-8"))
        h.update(b"|")
        if position is not None:
            h.update(repr(position).encode("utf-8"))
        h.update(b"|")
        if rotation is not None:
            h.update(repr(rotation).encode("utf-8"))
        h.update(b"|")
        if image is not None:
            h.update(str(id(image)).encode("ascii"))
        return h.hexdigest()

    @staticmethod
    def _to_1d_tensor(x: Union[torch.Tensor, List[float], Tuple[float, ...]], name: str) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.ndim != 1:
            raise ValueError(f"{name} must be 1D, got shape={tuple(t.shape)}")
        return t

    @staticmethod
    def _to_2d_tensor(x: Union[torch.Tensor, List[List[float]]], name: str) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape={tuple(t.shape)}")
        return t

    @staticmethod
    def _to_3d_tensor(x: Union[torch.Tensor, List[List[List[float]]]], name: str) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.ndim != 3:
            raise ValueError(f"{name} must be 3D, got shape={tuple(t.shape)}")
        return t

    @staticmethod
    def _to_4d_tensor(x: Union[torch.Tensor, List[List[List[List[float]]]]], name: str) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.ndim != 4:
            raise ValueError(f"{name} must be 4D, got shape={tuple(t.shape)}")
        return t

    @staticmethod
    def _cosine_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        a: [N, D], b: [D]
        return: [N]
        """
        a_n = a / (a.norm(dim=1, keepdim=True) + eps)
        b_n = b / (b.norm() + eps)
        return torch.matmul(a_n, b_n)

    def score_all(
        self,
        query_position: Union[torch.Tensor, List[float], Tuple[float, float, float]],
        query_instruction_emb: Union[torch.Tensor, List[float], Tuple[float, ...]],
        query_image_semantic_emb: Union[torch.Tensor, List[float], Tuple[float, ...]],
        query_image_query_emb: Union[torch.Tensor, List[float], Tuple[float, ...]],
        kb_scene_semantic_emb: Union[torch.Tensor, List[List[float]]],
        kb_zone_semantic_emb: Union[torch.Tensor, List[List[List[float]]]],
        kb_view_semantic_emb: Union[torch.Tensor, List[List[List[List[float]]]]],
        kb_view_image_emb: Union[torch.Tensor, List[List[List[List[float]]]]],
        kb_view_position: Union[torch.Tensor, List[List[List[List[float]]]]],
        cache_key: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Batch-score all items (torch) and write to the score cache.

        Input shape conventions (hierarchical tensors):
        - query_instruction_emb: [D] or [1, D]
        - query_image_semantic_emb: [D] or [1, D]
        - query_image_query_emb: [D] or [1, D]
        - kb_scene_semantic_emb: [S, D]
        - kb_zone_semantic_emb: [S, Z, D] (zero-padded when shorter)
        - kb_view_semantic_emb: [S, Z, V, D] (zero-padded when shorter)
        - kb_view_image_emb: [S, Z, V, D] (zero-padded when shorter)
        - kb_view_position: [S, Z, V, 3] (zero-padded when shorter)

        Main score fields returned:
        - view_image_similarity: [S, Z, V]
        - view_semantic_similarity: [S, Z, V] (view instruction semantics: KB view description vs query instruction)
        - view_image_semantic_similarity: [S, Z, V] (view image semantics: KB view description vs query image-caption text)
        - zone_image_semantic_similarity: [S, Z] (zone image semantics: KB zone description vs query image-caption text)
        - scene_img_semantic_similarity: [S]
        - scene_belonging_score: alpha*scene image semantics + (1-alpha)*max view image similarity per scene
        - start_zone_belonging_score: [S, Z, 1] = scene_belonging_score * zone image semantics
        - start_view_position_score: [S, Z, V, 1] = scene_belonging_score * [ beta*distance + (1-beta)*(alpha*view image semantics + (1-alpha)*view image) ]
        - end_zone_belonging_score: [S, Z, 1] = scene_belonging_score * zone_image_semantic_similarity (end zone from end view; kept for compatibility)
        - end_view_score: [S, Z, V, 1] = scene_belonging_score * view instruction semantics (optionally normalized)

        When ``normalize_visual_semantic_scores`` (YAML ``retrieval.normalize_visual_semantic_scores``) is true,
        scene/zone/view **visual and semantic cosines** are normalized via ``score_norm_mode`` (minmax/zscore) before alpha mixing;
        **distance_score is not normalized**. Default off when the key is omitted (legacy behavior).
        Note: zero-padding (all-zero vectors) is masked as invalid automatically.
        """
        # Mix coefficients:
        # - alpha: text/image weight (scene match & start-view match)
        # - beta: position/view weight (start and end view match)
        alpha = float(getattr(self, "_alpha", 0.0))
        beta = float(getattr(self, "_beta", 0.0))


        r_pos = self._to_1d_tensor(query_position, "query_position")
        r_txt_t = torch.as_tensor(query_instruction_emb, dtype=torch.float32)
        r_img_sem_t = torch.as_tensor(query_image_semantic_emb, dtype=torch.float32)
        r_img_query_t = torch.as_tensor(query_image_query_emb, dtype=torch.float32)
        r_txt = r_txt_t.squeeze(0) if r_txt_t.ndim == 2 and r_txt_t.shape[0] == 1 else r_txt_t
        r_img_sem = (
            r_img_sem_t.squeeze(0) if r_img_sem_t.ndim == 2 and r_img_sem_t.shape[0] == 1 else r_img_sem_t
        )
        r_img_query = (
            r_img_query_t.squeeze(0)
            if r_img_query_t.ndim == 2 and r_img_query_t.shape[0] == 1
            else r_img_query_t
        )
        if r_txt.ndim != 1 or r_img_sem.ndim != 1 or r_img_query.ndim != 1:
            raise ValueError(
                "query_instruction_emb/query_image_semantic_emb/query_image_query_emb must be [D] or [1, D]"
            )

        scene_e = self._to_2d_tensor(kb_scene_semantic_emb, "kb_scene_semantic_emb")
        zone_e = self._to_3d_tensor(kb_zone_semantic_emb, "kb_zone_semantic_emb")
        view_e = self._to_4d_tensor(kb_view_semantic_emb, "kb_view_semantic_emb")
        view_img_e = self._to_4d_tensor(kb_view_image_emb, "kb_view_image_emb")
        view_pos = self._to_4d_tensor(kb_view_position, "kb_view_position")

        num_scene = scene_e.shape[0]
        if zone_e.shape[0] != num_scene:
            raise ValueError("kb_zone_semantic_emb first dim must match kb_scene_semantic_emb first dim")
        if view_e.shape[:2] != zone_e.shape[:2]:
            raise ValueError("kb_view_semantic_emb first 2 dims must match kb_zone_semantic_emb first 2 dims")
        if view_img_e.shape != view_e.shape:
            raise ValueError("kb_view_image_emb shape must equal kb_view_semantic_emb shape")
        if view_pos.shape[:3] != view_e.shape[:3]:
            raise ValueError("kb_view_position first 3 dims must match kb_view_semantic_emb first 3 dims")
        if (
            scene_e.shape[1] != r_txt.numel()
            or scene_e.shape[1] != r_img_sem.numel()
            or scene_e.shape[1] != r_img_query.numel()
        ):
            raise ValueError("embedding dim mismatch: query/text/image and KB embeddings must share same D")
        if zone_e.shape[2] != scene_e.shape[1] or view_e.shape[3] != scene_e.shape[1]:
            raise ValueError("zone/view embedding dim must match scene embedding dim")
        if view_pos.shape[3] != r_pos.numel():
            raise ValueError("kb_view_position last dim must match query_position dim")

        eps = 1e-8

        # 1) View image similarity (only when view image embedding is valid)
        view_img_norm = view_img_e.norm(dim=3)
        view_image_valid = view_img_norm > eps
        view_img_e_n = view_img_e / (view_img_norm.unsqueeze(-1) + eps)
        r_img_query_n = r_img_query / (r_img_query.norm() + eps)
        view_image_similarity = torch.sum(
            view_img_e_n * r_img_query_n.view(1, 1, 1, -1),
            dim=3,
        )
        view_image_similarity = torch.where(
            view_image_valid, view_image_similarity, torch.zeros_like(view_image_similarity)
        )

        # 2) View semantic similarity (semantic validity defines view existence; avoids dropping views without rendered images)
        view_sem_norm = view_e.norm(dim=3)
        view_sem_valid = view_sem_norm > eps
        view_e_n = view_e / (view_sem_norm.unsqueeze(-1) + eps)
        r_txt_n = r_txt / (r_txt.norm() + eps)
        view_semantic_similarity = torch.sum(
            view_e_n * r_txt_n.view(1, 1, 1, -1),
            dim=3,
        )
        view_semantic_similarity = torch.where(
            view_sem_valid,
            view_semantic_similarity,
            torch.zeros_like(view_semantic_similarity),
        )

        # View image semantics: view_semantic_emb (KB view description) vs query_image_semantic_emb (caption->text)
        r_img_sem_n = r_img_sem / (r_img_sem.norm() + eps)
        view_image_semantic_similarity = torch.sum(
            view_e_n * r_img_sem_n.view(1, 1, 1, -1),
            dim=3,
        )
        view_image_semantic_similarity = torch.where(
            view_sem_valid,
            view_image_semantic_similarity,
            torch.zeros_like(view_image_semantic_similarity),
        )

        # view_valid uses semantic validity (not whether view image emb exists)
        view_valid = view_sem_valid
        # 3) Zone semantics
        zone_norm = zone_e.norm(dim=2)
        zone_valid = zone_norm > 1e-8
        zone_e_n = zone_e / (zone_norm.unsqueeze(-1) + 1e-8)
        # Zone image semantics: zone_semantic_emb (KB zone description) vs query_image_semantic_emb (caption->text)
        zone_image_semantic_similarity = torch.sum(zone_e_n * r_img_sem_n.view(1, 1, -1), dim=2)
        zone_image_semantic_similarity = torch.where(
            zone_valid,
            zone_image_semantic_similarity,
            torch.zeros_like(zone_image_semantic_similarity),
        )
        # Scene image semantics: scene_semantic_emb (KB scene description) vs query_image_semantic_emb (caption->text)
        scene_img_semantic_similarity = self._cosine_matrix(scene_e, r_img_sem)

        # scene_max_view_image_sim[s] = max_{view in scene s}(view_image_similarity)
        vsim_masked = torch.where(
            view_image_valid, view_image_similarity, torch.full_like(view_image_similarity, -1e9)
        )
        scene_max_view_image_sim = vsim_masked.amax(dim=(1, 2))
        scene_has_view = view_image_valid.any(dim=(1, 2))
        scene_max_view_image_sim = torch.where(scene_has_view, scene_max_view_image_sim, torch.zeros_like(scene_max_view_image_sim))

        # Optional normalization for visual/semantic cosines (flag + score_norm_mode); distance is not normalized.
        base_norm_mode = str(getattr(self, "_score_norm_mode", "none"))
        norm_eps = float(getattr(self, "_score_norm_eps", 1e-6))
        norm_mode = base_norm_mode if bool(getattr(self, "_normalize_visual_semantic_scores", False)) else "none"
        scene_sem_for_mix = self._normalize_scores(scene_img_semantic_similarity, norm_mode, norm_eps)
        scene_vis_for_mix = self._normalize_scores(scene_max_view_image_sim, norm_mode, norm_eps)
        zone_sem_for_mix = self._normalize_scores(zone_image_semantic_similarity, norm_mode, norm_eps)
        view_img_sem_for_mix = self._normalize_scores(view_image_semantic_similarity, norm_mode, norm_eps)
        view_img_for_mix = self._normalize_scores(view_image_similarity, norm_mode, norm_eps)
        view_txt_sem_for_mix = self._normalize_scores(view_semantic_similarity, norm_mode, norm_eps)

        # Scene match: a*(scene image semantics) + (1-a)*(max view image similarity)
        scene_belonging_score = alpha * scene_sem_for_mix + (1 - alpha) * scene_vis_for_mix

        # Start/end zone belonging scores
        scene_score_2d = scene_belonging_score.view(num_scene, 1)
        # Start zone: scene_score * zone image semantics (normalized per flag)
        start_zone_belonging_score = (scene_score_2d * zone_sem_for_mix).unsqueeze(-1)
        # End zone is determined by end view; end_zone_belonging_score kept for output compatibility only.
        end_zone_belonging_score = (scene_score_2d * zone_sem_for_mix).unsqueeze(-1)

        # Distance score: 1 / (1 + ||view_pos - robot_pos||2)
        dist = torch.norm(view_pos - r_pos.view(1, 1, 1, -1), dim=3)
        distance_score = 1.0 / (1.0 + dist)
        distance_score = torch.where(view_valid, distance_score, torch.zeros_like(distance_score))

        scene_score_3d = scene_belonging_score.view(num_scene, 1, 1)
        view_scene_gate = (
            torch.ones_like(scene_score_3d)
            if bool(getattr(self, "_no_scene_gate_on_view_scores", False))
            else scene_score_3d
        )
        # Start-view match score:
        # scene_score * [ beta*distance + (1-beta)*( alpha*view image semantics + (1-alpha)*view image ) ]
        start_view_position_score = (
            view_scene_gate
            * (
                beta * distance_score
                + (1 - beta)
                * (
                    alpha * view_img_sem_for_mix
                    + (1 - alpha) * view_img_for_mix
                )
            )
        ).unsqueeze(-1)

        # End-view match score (semantics only, no distance):
        # scene_score * view instruction semantics (normalized per flag)
        end_view_score = (
            view_scene_gate
            * view_txt_sem_for_mix
        ).unsqueeze(-1)

        out: Dict[str, torch.Tensor] = {
            "view_image_similarity": view_image_similarity,
            "view_semantic_similarity": view_semantic_similarity,
            "view_image_semantic_similarity": view_image_semantic_similarity,
            "zone_image_semantic_similarity": zone_image_semantic_similarity,
            "scene_img_semantic_similarity": scene_img_semantic_similarity,
            "scene_max_view_image_similarity": scene_max_view_image_sim,
            "scene_img_semantic_similarity_for_mix": scene_sem_for_mix,
            "scene_max_view_image_similarity_for_mix": scene_vis_for_mix,
            "zone_image_semantic_similarity_for_mix": zone_sem_for_mix,
            "view_semantic_similarity_for_mix": view_txt_sem_for_mix,
            "scene_belonging_score": scene_belonging_score.view(num_scene, 1),
            "distance_score": distance_score,
            "start_zone_belonging_score": start_zone_belonging_score,
            "start_view_position_score": start_view_position_score,
            "end_zone_belonging_score": end_zone_belonging_score,
            "end_view_score": end_view_score,
            "zone_valid_mask": zone_valid,
            "view_valid_mask": view_valid,
        }

        if self._enable_score_cache:
            k = cache_key or self._fingerprint_query(
                instruction=(
                    f"tensor:{tuple(r_txt.shape)}|alpha:{float(getattr(self, '_alpha', 0.0))}"
                    f"|beta:{float(getattr(self, '_beta', 0.0))}|vsem_norm:{int(bool(getattr(self, '_normalize_visual_semantic_scores', False)))}"
                    f"|snm:{norm_mode}|nosg:{int(bool(getattr(self, '_no_scene_gate_on_view_scores', False)))}"
                ),
                image=None,
                position=tuple(float(v) for v in r_pos.tolist()),
                rotation=None,
            )
            self._score_cache.set(("score_all", k), out)
        return out

    def retrieve(
        self,
        kb: KnowledgeBase,
        *,
        instruction: str,
        robot_position: Union[torch.Tensor, List[float], Tuple[float, float, float]],
        robot_image: Any = None,
        topk1_scenes: int = 1,
        topk2_zones: int = 2,
        topk3_pairs: int = 3,
        pair_ranking_topk: Optional[int] = None,
        embed_view_images: bool = True,
        verbose: bool = True,
        timing: bool = False,
        progress: bool = False,
        kb_cache_path: Optional[Union[str, Path]] = None,
        force_rebuild_kb_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry: given KB + state, return navigation retrieval plan JSON.

        Steps:
        1) Embed KB (scene/zone/view text + view images).
        2) `score_all` for scene/zone/start-end scores and masks.
        3) Rerank: top-K scenes, start zones, start-end pairs.
        4) Dijkstra on `kb` view_graph.adjacency for each pair path.
        5) Return JSON with `robot_caption` (VLM English when dual embedders + `robot_image`, else `None`),
           `topk1_scenes` / `topk2_zones` / `topk3_pairs`.

        Optional ``pair_ranking_topk``: when not ``None``, overrides ``topk3_pairs`` for per-scene start/end
        candidate depth and final pair list length (longer lists for MRR eval; 1/rank, no rank>k zeroing).

        When ``timing=True``, also return ``timing_ms`` (ms) and print:
        ``total`` / ``llm_call`` /
        ``text_embedding_query`` / ``text_embedding_kb`` /
        ``image_embedding_query`` / ``image_embedding_kb`` /
        ``scoring`` / ``output_with_path``.
        """
        def _v(msg: str) -> None:
            if verbose:
                print(msg, flush=True)

        _v("[retrieve] start")
        _v(f"[retrieve] instruction={instruction!r}, robot_position={robot_position}")

        def _maybe_sync_cuda() -> None:
            if torch is None:
                return
            if torch.cuda.is_available():
                # Sync CUDA before timing so async work is not under-counted
                torch.cuda.synchronize()

        t_total_0 = time.perf_counter() if timing else 0.0
        timings_s: Dict[str, float] = {}
        matrix_occupancy_stats: Dict[str, Dict[str, Any]] = {}

        def _add(name: str, dt: float) -> None:
            if timing and dt > 0.0:
                timings_s[name] = timings_s.get(name, 0.0) + float(dt)

        def _record_tensor_occupancy(name: str, tensor: Optional[torch.Tensor]) -> None:
            if tensor is None:
                return
            t = torch.as_tensor(tensor)
            numel = int(t.numel())
            if numel <= 0:
                matrix_occupancy_stats[name] = {
                    "shape": list(t.shape),
                    "numel": 0,
                    "occupied_count": 0,
                    "occupied_ratio": 0.0,
                    "memory_mb": 0.0,
                }
                return
            if t.dtype == torch.bool:
                occupied_count = int(t.sum().item())
            elif t.is_floating_point():
                occupied_count = int((t != 0).sum().item())
            else:
                occupied_count = int((t != 0).sum().item())
            occupied_ratio = float(occupied_count) / float(numel)
            memory_mb = float(numel * t.element_size()) / (1024.0 * 1024.0)
            matrix_occupancy_stats[name] = {
                "shape": list(t.shape),
                "numel": numel,
                "occupied_count": occupied_count,
                "occupied_ratio": occupied_ratio,
                "memory_mb": memory_mb,
            }

        def _finalize_timing() -> Dict[str, float]:
            if not timing:
                return {}
            total = time.perf_counter() - t_total_0
            timings_s["total_wall"] = total
            for _zkey in (
                "vlm_caption",
                "query_instruction_emb",
                "query_caption_text_emb",
                "query_robot_image_emb",
                "kb_text_emb",
                "kb_view_image_emb",
                "score_all",
                "topk_scene_zone",
                "view_graph_prep",
                "pair_enum_dijkstra",
                "postprocess_output",
            ):
                timings_s.setdefault(_zkey, 0.0)
            llm_call = timings_s.get("vlm_caption", 0.0)
            text_embedding_query = (
                timings_s.get("query_instruction_emb", 0.0)
                + timings_s.get("query_caption_text_emb", 0.0)
            )
            text_embedding_kb = timings_s.get("kb_text_emb", 0.0)
            image_embedding_query = (
                timings_s.get("query_robot_image_emb", 0.0)
            )
            image_embedding_kb = timings_s.get("kb_view_image_emb", 0.0)
            scoring = timings_s.get("score_all", 0.0)
            output_with_path = (
                timings_s.get("topk_scene_zone", 0.0)
                + timings_s.get("view_graph_prep", 0.0)
                + timings_s.get("pair_enum_dijkstra", 0.0)
                + timings_s.get("postprocess_output", 0.0)
            )

            timing_ms = {
                "total": round(total * 1000.0, 3),
                "llm_call": round(llm_call * 1000.0, 3),
                "text_embedding_query": round(text_embedding_query * 1000.0, 3),
                "text_embedding_kb": round(text_embedding_kb * 1000.0, 3),
                "image_embedding_query": round(image_embedding_query * 1000.0, 3),
                "image_embedding_kb": round(image_embedding_kb * 1000.0, 3),
                "scoring": round(scoring * 1000.0, 3),
                "output_with_path": round(output_with_path * 1000.0, 3),
            }
            print("[timing] --- metrics (ms) ---", flush=True)
            for k in (
                "total",
                "llm_call",
                "text_embedding_query",
                "text_embedding_kb",
                "image_embedding_query",
                "image_embedding_kb",
                "scoring",
                "output_with_path",
            ):
                print(f"  {k}: {timing_ms[k]:.3f}", flush=True)
            if matrix_occupancy_stats:
                print("[timing] --- matrix occupancy ---", flush=True)
                for key in (
                    "kb_scene_semantic_emb",
                    "kb_zone_semantic_emb",
                    "kb_view_semantic_emb",
                    "kb_view_image_emb",
                    "kb_view_position",
                    "zone_valid_mask",
                    "view_valid_mask",
                    "start_view_position_score",
                    "end_view_score",
                ):
                    st = matrix_occupancy_stats.get(key)
                    if st is None:
                        continue
                    print(
                        "  "
                        f"{key}: shape={st['shape']} numel={st['numel']} "
                        f"occupied={st['occupied_count']} ({st['occupied_ratio'] * 100.0:.2f}%) "
                        f"memory={st['memory_mb']:.2f}MB",
                        flush=True,
                    )
            return timing_ms

        if self._text_embedder is None or self._vision_embedder is None:
            raise NotImplementedError("set text_embedder + vision_embedder")

        # -------------------------
        # 0) Build scene/zone/view indices
        # -------------------------
        def _is_valid_scene_id(scene_id: str) -> bool:
            safe = "".join(c for c in scene_id if c.isalnum() or c in "-_")
            return safe == scene_id

        z_cnt_max = 0
        v_cnt_max = 0

        scene_ids = [sid for sid in kb.list_scene_ids() if _is_valid_scene_id(sid)]
        s_cnt = len(scene_ids)
        _v(f"[retrieve] valid scene count={s_cnt}")
        if s_cnt == 0:
            out_empty: Dict[str, Any] = {
                "topk1_scenes": [],
                "topk2_zones": [],
                "topk3_pairs": [],
                "robot_caption": None,
            }
            if timing:
                out_empty["timing_ms"] = _finalize_timing()
            return out_empty

        # Per scene:
        # - zone_ids sorted by scene.attributes.zone_ids
        # - view_ids sorted by scene.attributes.view_ids (for adjacency index mapping)
        scene_zone_ids: List[List[str]] = []
        scene_view_ids_order: List[List[str]] = []
        view_ids_in_zone: List[List[List[str]]] = []  # [S][Z][Vi]
        z_cnts: List[int] = []
        v_cnt_max = 0

        t_ix0 = time.perf_counter() if timing else 0.0
        for sid in scene_ids:
            tree = kb.scene(sid)
            scene_attr = (tree.get("scene") or {}).get("attributes") or {}
            zone_ids = scene_attr.get("zone_ids") or list((tree.get("zones") or {}).keys())
            view_ids_order = scene_attr.get("view_ids") or list((tree.get("views") or {}).keys())

            zones_dict = tree.get("zones") or {}
            views_dict = tree.get("views") or {}
            # view->zone mapping to bucket views by zone
            view_to_zone: Dict[str, Optional[str]] = {}
            for vid, vnode in views_dict.items():
                attrs = vnode.get("attributes") or {}
                view_to_zone[str(vid)] = attrs.get("zone_id")

            per_scene_zone_views: List[List[str]] = []
            for zid in zone_ids:
                vids = [vid for vid in view_ids_order if view_to_zone.get(str(vid)) == zid]
                per_scene_zone_views.append(list(map(str, vids)))
                v_cnt_max = max(v_cnt_max, len(vids))

            scene_zone_ids.append(list(map(str, zone_ids)))
            scene_view_ids_order.append(list(map(str, view_ids_order)))
            view_ids_in_zone.append(per_scene_zone_views)
            z_cnts.append(len(zone_ids))

        if timing:
            _add("index_structure", time.perf_counter() - t_ix0)

        z_cnt_max = max(z_cnts) if z_cnts else 0
        _v(f"[retrieve] z_cnt_max={z_cnt_max}, v_cnt_max={v_cnt_max}")
        if z_cnt_max <= 0:
            out_empty2: Dict[str, Any] = {
                "topk1_scenes": [],
                "topk2_zones": [],
                "topk3_pairs": [],
                "robot_caption": None,
            }
            if timing:
                out_empty2["timing_ms"] = _finalize_timing()
            return out_empty2

        # When robot_image is set, captioner produces query image semantic text
        robot_caption: Optional[str] = None

        # -------------------------
        # 1) Compute embeddings (scene/zone/view text + view images)
        # -------------------------
        def as_1d_float_tensor(x: Any, name: str) -> torch.Tensor:
            t = torch.as_tensor(x, dtype=torch.float32)
            if t.ndim != 1:
                t = t.flatten()
            if t.numel() < 1:
                raise ValueError(f"{name} embedding is empty")
            return t

        assert self._text_embedder is not None and self._vision_embedder is not None
        te = self._text_embedder
        ve = self._vision_embedder

        def text_vec(s: str) -> torch.Tensor:
            r = te.embed(instruction=str(s), image=None, position=None, rotation=None)
            return as_1d_float_tensor(r.get("text_emb"), "text_emb")

        def view_image_vec(im: Any) -> torch.Tensor:
            r = ve.embed(
                instruction="",
                image=im,
                position=None,
                rotation=None,
                image_role="kb_view",
            )
            return as_1d_float_tensor(r.get("image_feat"), "view image_feat")

        t_q0 = time.perf_counter() if timing else 0.0
        if timing:
            _maybe_sync_cuda()
        # BGE retrieval: query prefix on user instruction only; not on caption or KB text
        instr_for_text_emb = instruction
        if (
            isinstance(te, BGEEmbedder)
            and self._bge_query_prefix
            and instruction is not None
            and str(instruction).strip()
        ):
            instr_for_text_emb = self._bge_query_prefix + str(instruction)
        _v(
            "[retrieve] text_embedder instruction input: "
            f"{instr_for_text_emb!r} "
            f"(raw_instruction={instruction!r}, bge={isinstance(te, BGEEmbedder)})"
        )
        query_instruction_emb = text_vec(instr_for_text_emb)
        if timing:
            _maybe_sync_cuda()
            _add("query_instruction_emb", time.perf_counter() - t_q0)
        d_dim = int(te.embedding_dim)
        _v(f"[retrieve] embedding dim D={d_dim} (text_embedder + vision_embedder)")

        if robot_image is not None:
            assert self._robot_captioner is not None
            _v("[retrieve] VLM caption for robot observation (DashScope / retrieval.caption) ...")
            t_cap0 = time.perf_counter() if timing else 0.0
            cap = self._robot_captioner.caption(robot_image)
            if timing:
                _add("vlm_caption", time.perf_counter() - t_cap0)
            robot_caption = cap
            _v(f"[retrieve] caption -> text_embedder: {cap[:280]!r}...")
            t_cs0 = time.perf_counter() if timing else 0.0
            if timing:
                _maybe_sync_cuda()
            query_image_semantic_emb = text_vec(cap)
            if timing:
                _maybe_sync_cuda()
                _add("query_caption_text_emb", time.perf_counter() - t_cs0)

            t_rv0 = time.perf_counter() if timing else 0.0
            if timing:
                _maybe_sync_cuda()
            r = ve.embed(
                instruction="",
                image=robot_image,
                position=None,
                rotation=None,
                image_role="robot",
            )
            query_image_query_emb = as_1d_float_tensor(r.get("image_feat"), "robot image_feat")
            if timing:
                _maybe_sync_cuda()
                _add("query_robot_image_emb", time.perf_counter() - t_rv0)
        else:
            query_image_semantic_emb = torch.zeros(d_dim, dtype=torch.float32)
            query_image_query_emb = torch.zeros(d_dim, dtype=torch.float32)
        if int(query_image_semantic_emb.numel()) != d_dim:
            raise ValueError("text_emb/image_feat dim mismatch")

        cache_key = self._kb_cache_key(
            kb=kb,
            scene_ids=scene_ids,
            z_cnt_max=z_cnt_max,
            v_cnt_max=v_cnt_max,
            d_dim=d_dim,
            embed_view_images=embed_view_images,
        )
        kb_cache_file = Path(kb_cache_path) if kb_cache_path is not None else None
        cache_payload: Optional[Dict[str, Any]] = None
        if not force_rebuild_kb_cache:
            cache_payload = self._kb_embed_cache_mem.get(cache_key)
            if cache_payload is None and kb_cache_file is not None:
                cache_payload = self._load_kb_cache_file(kb_cache_file)
                if isinstance(cache_payload, dict) and cache_payload.get("cache_key") != cache_key:
                    cache_payload = None

        if cache_payload is not None:
            _v("[retrieve] using cached KB embeddings")
            kb_scene_semantic_emb = torch.as_tensor(cache_payload["kb_scene_semantic_emb"], dtype=torch.float32)
            kb_zone_semantic_emb = torch.as_tensor(cache_payload["kb_zone_semantic_emb"], dtype=torch.float32)
            kb_view_semantic_emb = torch.as_tensor(cache_payload["kb_view_semantic_emb"], dtype=torch.float32)
            kb_view_image_emb = torch.as_tensor(cache_payload["kb_view_image_emb"], dtype=torch.float32)
            kb_view_position = torch.as_tensor(cache_payload["kb_view_position"], dtype=torch.float32)
            zone_id_map = cache_payload["zone_id_map"]
            view_id_map = cache_payload["view_id_map"]
        else:
            # Preallocate padding tensors + id maps (can use noticeable memory)
            t_kprep0 = time.perf_counter() if timing else 0.0
            kb_scene_semantic_emb = torch.zeros((s_cnt, d_dim), dtype=torch.float32)
            kb_zone_semantic_emb = torch.zeros((s_cnt, z_cnt_max, d_dim), dtype=torch.float32)
            kb_view_semantic_emb = torch.zeros((s_cnt, z_cnt_max, v_cnt_max, d_dim), dtype=torch.float32)
            kb_view_image_emb = torch.zeros((s_cnt, z_cnt_max, v_cnt_max, d_dim), dtype=torch.float32)
            kb_view_position = torch.zeros((s_cnt, z_cnt_max, v_cnt_max, 3), dtype=torch.float32)
            _v(
                "[retrieve] allocated tensors: "
                f"kb_scene_semantic_emb={tuple(kb_scene_semantic_emb.shape)}, "
                f"kb_zone_semantic_emb={tuple(kb_zone_semantic_emb.shape)}, "
                f"kb_view_semantic_emb={tuple(kb_view_semantic_emb.shape)}"
            )

            # Map (s,z,v) indices to KB ids
            zone_id_map: List[List[Optional[str]]] = [[None for _ in range(z_cnt_max)] for _ in range(s_cnt)]
            view_id_map: List[List[List[Optional[str]]]] = [
                [[None for _ in range(v_cnt_max)] for _ in range(z_cnt_max)] for _ in range(s_cnt)
            ]
            if timing:
                _add("kb_tensor_and_maps", time.perf_counter() - t_kprep0)

            scene_iter = enumerate(scene_ids)
            if progress and tqdm is not None:
                scene_iter = enumerate(
                    tqdm(scene_ids, desc="Embed KB scenes", unit="scene")
                )
            for s_idx, sid in scene_iter:
                tree = kb.scene(sid)
                scene_attr = (tree.get("scene") or {}).get("attributes") or {}

                # scene text embedding
                s_desc = scene_attr.get("description", "") or ""
                t_ks = time.perf_counter() if timing else 0.0
                if timing:
                    _maybe_sync_cuda()
                kb_scene_semantic_emb[s_idx] = text_vec(str(s_desc))[:d_dim]
                if timing:
                    _maybe_sync_cuda()
                    _add("kb_text_emb", time.perf_counter() - t_ks)

                zone_ids = scene_zone_ids[s_idx]
                per_scene_views_in_zone = view_ids_in_zone[s_idx]
                zones_dict = tree.get("zones") or {}
                views_dict = tree.get("views") or {}

                for z_idx, zid in enumerate(zone_ids):
                    if z_idx >= z_cnt_max:
                        break
                    zone_id_map[s_idx][z_idx] = zid

                    z_attrs = (zones_dict.get(zid) or {}).get("attributes") or {}
                    z_desc = z_attrs.get("description", "") or ""
                    t_kz = time.perf_counter() if timing else 0.0
                    if timing:
                        _maybe_sync_cuda()
                    kb_zone_semantic_emb[s_idx, z_idx] = text_vec(str(z_desc))[:d_dim]
                    if timing:
                        _maybe_sync_cuda()
                        _add("kb_text_emb", time.perf_counter() - t_kz)

                    # view embedding for this zone
                    view_ids_order_in_zone = per_scene_views_in_zone[z_idx]
                    for v_idx, vid in enumerate(view_ids_order_in_zone):
                        if v_idx >= v_cnt_max:
                            break
                        view_id_map[s_idx][z_idx][v_idx] = vid
                        v_attrs = (views_dict.get(vid) or {}).get("attributes") or {}
                        v_desc = v_attrs.get("description", "") or ""
                        t_kv = time.perf_counter() if timing else 0.0
                        if timing:
                            _maybe_sync_cuda()
                        kb_view_semantic_emb[s_idx, z_idx, v_idx] = text_vec(str(v_desc))[:d_dim]
                        if timing:
                            _maybe_sync_cuda()
                            _add("kb_text_emb", time.perf_counter() - t_kv)

                        pos = v_attrs.get("position") or [0.0, 0.0, 0.0]
                        if isinstance(pos, list) and len(pos) == 3:
                            kb_view_position[s_idx, z_idx, v_idx] = torch.as_tensor(pos, dtype=torch.float32)

                        if embed_view_images:
                            t_kimg = time.perf_counter() if timing else 0.0
                            img = kb.load_view_image(sid, vid)
                            if img is not None:
                                if timing:
                                    _maybe_sync_cuda()
                                kb_view_image_emb[s_idx, z_idx, v_idx] = view_image_vec(img)[:d_dim]
                                if timing:
                                    _maybe_sync_cuda()
                            if timing:
                                _add("kb_view_image_emb", time.perf_counter() - t_kimg)

            payload = {
                "cache_key": cache_key,
                "kb_scene_semantic_emb": kb_scene_semantic_emb.cpu(),
                "kb_zone_semantic_emb": kb_zone_semantic_emb.cpu(),
                "kb_view_semantic_emb": kb_view_semantic_emb.cpu(),
                "kb_view_image_emb": kb_view_image_emb.cpu(),
                "kb_view_position": kb_view_position.cpu(),
                "zone_id_map": zone_id_map,
                "view_id_map": view_id_map,
            }
            self._kb_embed_cache_mem[cache_key] = payload
            if kb_cache_file is not None:
                self._save_kb_cache_file(kb_cache_file, payload)

        # -------------------------
        # 2) Batch score via score_all
        # -------------------------
        _v("[retrieve] embeddings filled, calling score_all() ...")
        query_position_t = torch.as_tensor(robot_position, dtype=torch.float32).flatten()
        if query_position_t.numel() != 3:
            raise ValueError("robot_position must be [x,y,z]")

        t_sc0 = time.perf_counter() if timing else 0.0
        if timing:
            _maybe_sync_cuda()
        score_out = self.score_all(
            query_position=query_position_t,
            query_instruction_emb=query_instruction_emb,
            # Scene match: image semantic query (caption->text embed)
            query_image_semantic_emb=query_image_semantic_emb,
            # View image match: direct ViT image_feat query
            query_image_query_emb=query_image_query_emb,
            kb_scene_semantic_emb=kb_scene_semantic_emb,
            kb_zone_semantic_emb=kb_zone_semantic_emb,
            kb_view_semantic_emb=kb_view_semantic_emb,
            kb_view_image_emb=kb_view_image_emb,
            kb_view_position=kb_view_position,
        )
        if timing:
            _maybe_sync_cuda()
            _add("score_all", time.perf_counter() - t_sc0)
        _v("[retrieve] score_all() finished")

        scene_scores = score_out["scene_belonging_score"].squeeze(-1)  # [S]
        start_zone_scores = score_out["start_zone_belonging_score"].squeeze(-1)  # [S,Z]
        start_view_scores = score_out["start_view_position_score"].squeeze(-1)  # [S,Z,V]
        end_view_scores = score_out["end_view_score"].squeeze(-1)  # [S,Z,V]
        zone_valid_mask = score_out["zone_valid_mask"]  # [S,Z]
        view_valid_mask = score_out["view_valid_mask"]  # [S,Z,V]
        if timing:
            _record_tensor_occupancy("kb_scene_semantic_emb", kb_scene_semantic_emb)
            _record_tensor_occupancy("kb_zone_semantic_emb", kb_zone_semantic_emb)
            _record_tensor_occupancy("kb_view_semantic_emb", kb_view_semantic_emb)
            _record_tensor_occupancy("kb_view_image_emb", kb_view_image_emb)
            _record_tensor_occupancy("kb_view_position", kb_view_position)
            _record_tensor_occupancy("zone_valid_mask", zone_valid_mask)
            _record_tensor_occupancy("view_valid_mask", view_valid_mask)
            _record_tensor_occupancy("start_view_position_score", start_view_scores)
            _record_tensor_occupancy("end_view_score", end_view_scores)

        # -------------------------
        # 3) rerank：topk1 scenes / topk2 zones / topk3 pairs
        # -------------------------
        t_tk0 = time.perf_counter() if timing else 0.0
        k1 = min(max(1, int(topk1_scenes)), s_cnt)
        k2 = min(max(1, int(topk2_zones)), z_cnt_max * s_cnt)
        k3 = min(max(1, int(topk3_pairs)), 10_000)  # pairs may be trimmed further
        k_pair = (
            min(max(1, int(pair_ranking_topk)), 10_000)
            if pair_ranking_topk is not None
            else k3
        )
        _v(f"[retrieve] rerank: k1={k1}, k2={k2}, k3={k3}, k_pair={k_pair}")

        topk1_vals, topk1_scene_idx = torch.topk(scene_scores, k=k1, dim=0)
        topk1_scene_idx_list = topk1_scene_idx.tolist()
        _v(f"[retrieve] topk1 scene idx={topk1_scene_idx_list}")

        # topk2 zones: global across scenes (no scene pre-filter)
        zone_scores_masked = torch.where(
            zone_valid_mask,
            start_zone_scores,
            torch.full_like(start_zone_scores, -1e9),
        )
        topk2_flat_vals, topk2_flat_idx = torch.topk(zone_scores_masked.flatten(), k=k2, dim=0)
        topk2_scene_idx = (topk2_flat_idx // z_cnt_max).tolist()
        topk2_zone_idx = (topk2_flat_idx % z_cnt_max).tolist()
        _v(f"[retrieve] topk2 pairs={list(zip(topk2_scene_idx, topk2_zone_idx))}")

        # For pairs: no zone pre-filter; compare starts across all valid views
        zone_sel_mask = zone_valid_mask
        if timing:
            _add("topk_scene_zone", time.perf_counter() - t_tk0)

        # topk3 pairs: enumerate top-starts x top-ends per scene, then global topk3
        pair_candidates: List[Dict[str, Any]] = []

        # Precompute per-scene zone_graph adjacency and zone_ids mapping
        scene_zone_adj: List[Optional[torch.Tensor]] = [None for _ in range(s_cnt)]
        scene_zone_ids_order: List[List[str]] = [[] for _ in range(s_cnt)]
        scene_zone_id_to_idx: List[Dict[str, int]] = [{} for _ in range(s_cnt)]

        t_gp0 = time.perf_counter() if timing else 0.0
        for s_idx, sid in enumerate(scene_ids):
            tree = kb.scene(sid)
            scene_attr = (tree.get("scene") or {}).get("attributes") or {}
            zone_ids_order = scene_attr.get("zone_ids") or list((tree.get("zones") or {}).keys())
            zone_graph = scene_attr.get("zone_graph") or {}
            adjacency = zone_graph.get("adjacency")
            if adjacency is None:
                continue
            scene_zone_adj[s_idx] = torch.as_tensor(adjacency, dtype=torch.float32)
            scene_zone_ids_order[s_idx] = list(map(str, zone_ids_order))
            scene_zone_id_to_idx[s_idx] = {zid: i for i, zid in enumerate(scene_zone_ids_order[s_idx])}
        if timing:
            _add("view_graph_prep", time.perf_counter() - t_gp0)

        # Enumerate pairs per scene globally (no scene pre-filter)
        t_pe0 = time.perf_counter() if timing else 0.0
        pair_scene_iter = range(s_cnt)
        if progress and tqdm is not None:
            pair_scene_iter = tqdm(pair_scene_iter, desc="Enumerate scene pairs", unit="scene")
        for s_i in pair_scene_iter:
            if scene_zone_adj[s_i] is None:
                continue
            adj = scene_zone_adj[s_i]
            _v(f"[retrieve] enumerating pairs in scene_idx={s_i} (scene_id={scene_ids[s_i]!r})")

            # start: valid views in the whole scene (no zone pre-filter)
            start_mask = view_valid_mask[s_i] & zone_sel_mask[s_i].view(z_cnt_max, 1)
            start_scores_s = start_view_scores[s_i]  # [Z,V]
            start_masked = torch.where(start_mask, start_scores_s, torch.full_like(start_scores_s, -1e9))
            start_flat = start_masked.flatten()
            valid_start_count = int((start_masked > -1e8).sum().item())
            if valid_start_count <= 0:
                _v(f"[retrieve] scene_idx={s_i}: no valid starts, skip")
                continue
            K_start = min(int(k_pair), valid_start_count)
            _v(f"[retrieve] scene_idx={s_i}: valid_start_count={valid_start_count}, K_start={K_start}")
            top_start_vals, top_start_flat_idx = torch.topk(start_flat, k=K_start, dim=0)

            # end: valid views in the whole scene
            end_mask = view_valid_mask[s_i]
            end_scores_s = end_view_scores[s_i]  # [Z,V]
            end_masked = torch.where(end_mask, end_scores_s, torch.full_like(end_scores_s, -1e9))
            end_flat = end_masked.flatten()
            valid_end_count = int((end_masked > -1e8).sum().item())
            if valid_end_count <= 0:
                _v(f"[retrieve] scene_idx={s_i}: no valid ends, skip")
                continue
            K_end = min(int(k_pair), valid_end_count)
            _v(f"[retrieve] scene_idx={s_i}: valid_end_count={valid_end_count}, K_end={K_end}")
            top_end_vals, top_end_flat_idx = torch.topk(end_flat, k=K_end, dim=0)

            Vmax = v_cnt_max
            # Enumerate start/end combinations
            for sv_val, sv_flat in zip(top_start_vals.tolist(), top_start_flat_idx.tolist()):
                z_start = int(sv_flat // Vmax)
                v_start = int(sv_flat % Vmax)
                start_view_id = view_id_map[s_i][z_start][v_start]
                if start_view_id is None:
                    continue
                start_zone_id = zone_id_map[s_i][z_start]
                s_idx_node = scene_zone_id_to_idx[s_i].get(start_zone_id) if start_zone_id is not None else None
                if s_idx_node is None or start_zone_id is None:
                    continue

                for ev_val, ev_flat in zip(top_end_vals.tolist(), top_end_flat_idx.tolist()):
                    z_end = int(ev_flat // Vmax)
                    v_end = int(ev_flat % Vmax)
                    end_view_id = view_id_map[s_i][z_end][v_end]
                    if end_view_id is None:
                        continue
                    end_zone_id = zone_id_map[s_i][z_end]
                    e_idx_node = scene_zone_id_to_idx[s_i].get(end_zone_id) if end_zone_id is not None else None
                    if e_idx_node is None or end_zone_id is None:
                        continue

                    pair_score = float(sv_val) * float(ev_val)
                    end_view_as_start_score = float(start_scores_s[z_end, v_end].item())
                    path_idx = dijkstra_shortest_path(int(s_idx_node), int(e_idx_node), adj)
                    path_zone_ids = [scene_zone_ids_order[s_i][pi] for pi in path_idx] if path_idx else []

                    pair_candidates.append(
                        {
                            "scene_id": scene_ids[s_i],
                            "start_zone_id": start_zone_id,
                            "start_view_id": start_view_id,
                            "end_zone_id": end_zone_id,
                            "end_view_id": end_view_id,
                            "scores": {
                                "start_view_position_score": float(sv_val),
                                "end_view_score": float(ev_val),
                                "end_view_as_start_score": end_view_as_start_score,
                                "pair_score": pair_score,
                            },
                            "path": path_zone_ids,
                            "path_zone_ids": path_zone_ids,
                        }
                    )

        pair_candidates.sort(key=lambda x: x["scores"]["pair_score"], reverse=True)
        pair_candidates = pair_candidates[:k_pair]
        _v(f"[retrieve] pair_candidates final={len(pair_candidates)}")
        if timing:
            _add("pair_enum_dijkstra", time.perf_counter() - t_pe0)

        # topk1 scenes and topk2 zones output (for debugging)
        t_po0 = time.perf_counter() if timing else 0.0
        topk1_scene_out = [
            {"scene_id": scene_ids[int(i)], "scene_belonging_score": float(v)}
            for v, i in zip(topk1_vals.tolist(), topk1_scene_idx.tolist())
        ]
        topk2_zone_out = []
        for s_i, z_i, v in zip(topk2_scene_idx, topk2_zone_idx, topk2_flat_vals.tolist()):
            zid = zone_id_map[s_i][z_i]
            if zid is None:
                continue
            topk2_zone_out.append(
                {
                    "scene_id": scene_ids[s_i],
                    "zone_id": zid,
                    "start_zone_belonging_score": float(v),
                }
            )

        if timing:
            _add("postprocess_output", time.perf_counter() - t_po0)

        out: Dict[str, Any] = {
            "robot_caption": robot_caption,
            "topk1_scenes": topk1_scene_out,
            "topk2_zones": topk2_zone_out,
            "topk3_pairs": pair_candidates,
        }
        if timing:
            out["timing_ms"] = _finalize_timing()
        return out

def dijkstra_shortest_path(
    start: int,
    end: int,
    adjacency: Union[torch.Tensor, List[List[float]]],
    *,
    no_edge_value: float = 0.0,
) -> List[int]:
    """
    Dijkstra shortest path (returns node indices along the path).

    Args:
        start: start node index (start view or zone node).
        end: end node index.
        adjacency: adjacency matrix [N, N]; adjacency[u][v] is edge weight u->v.
                   Values <= no_edge_value mean no edge.
        no_edge_value: threshold for no edge (default 0.0).

    Returns:
        nodes: node index list (e.g. [start, ..., end]); empty if unreachable.
    """
    if start == end:
        return [start]

    adj = torch.as_tensor(adjacency, dtype=torch.float32).detach().cpu()
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be [N,N], got shape={tuple(adj.shape)}")
    n = int(adj.shape[0])
    if not (0 <= start < n and 0 <= end < n):
        raise ValueError(f"start/end out of range for n={n}: start={start}, end={end}")

    # Dijkstra
    inf = float("inf")
    dist = [inf] * n
    prev: List[Optional[int]] = [None] * n
    dist[start] = 0.0

    pq: List[Tuple[float, int]] = [(0.0, start)]
    while pq:
        cur_d, u = heapq.heappop(pq)
        if cur_d != dist[u]:
            continue
        if u == end:
            break

        # Visit all neighbors (adjacency-matrix version)
        for v in range(n):
            w = float(adj[u, v].item())
            if v == u or w <= no_edge_value:
                continue
            nd = cur_d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if prev[end] is None:
        return []

    # Backtrack path
    path: List[int] = [end]
    cur = end
    while cur != start:
        p = prev[cur]
        if p is None:
            return []
        path.append(p)
        cur = p
    path.reverse()
    return path

