#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retriever-only evaluation for "Ours".

Metrics:
- Scene: Hit@1, Hit@K
- View(start/end): Hit@1, Hit@K, MRR（1/名次；GT 不在返回的 ``topk3_pairs`` 列表内为 0；列表长度由 ``--topk3`` 决定）

Each run writes under ``<result_dir>/<subset>_<timestamp>/``:
``ins_start_view/``, ``retriever_start_view/``, ``retriever_end_view/`` (see README).

Typical usage:
python rag4vln/scripts/eval/eval_retriever.py \
  --dataset-json data/vln_ce/raw_data/r2r/val_seen/val_seen.json \
  --gt-csv data/vln_ce/dataset_gt.csv \
  --subset-name full_instruction
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import numpy as np
import shutil
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _setup_sys_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    rag4vln_root = repo_root / "rag4vln"
    if str(rag4vln_root) not in sys.path:
        sys.path.insert(0, str(rag4vln_root))


def _cfg_embedding_dim(config_path: Path) -> int:
    from src.config_io import retrieval_embedding_dim  # noqa: E402

    try:
        return retrieval_embedding_dim(config_path)
    except Exception:
        return 768


def _scene_short_id(scene_id: Any) -> str:
    s = str(scene_id or "").strip()
    if not s:
        return ""
    # e.g. mp3d/s8pcmisQ38h/s8pcmisQ38h.glb -> s8pcmisQ38h
    parts = [p for p in s.split("/") if p]
    if len(parts) >= 2:
        return parts[-2]
    return Path(s).stem if "." in s else s


def _rank_in_list(items: List[Any], target: str, key: Optional[str] = None) -> Optional[int]:
    if not target:
        return None
    for i, row in enumerate(items):
        if key is None:
            v = str(row)
        else:
            if not isinstance(row, dict):
                continue
            v = str(row.get(key, ""))
        if v == target:
            return i + 1
    return None


def _hit_at(rank: Optional[int], k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def _rr(rank: Optional[int]) -> float:
    return 1.0 / float(rank) if rank is not None and rank > 0 else 0.0


def _avg(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _cosine(a: Any, b: Any, eps: float = 1e-8) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(va, vb) / (na * nb + eps))


def _view_position(kb: Any, scene_id: str, view_id: str) -> Optional[List[float]]:
    try:
        tree = kb.scene(scene_id)
    except Exception:
        return None
    views = (tree or {}).get("views") or {}
    attrs = (views.get(view_id) or {}).get("attributes") or {}
    pos = attrs.get("position")
    if isinstance(pos, list) and len(pos) == 3:
        try:
            return [float(pos[0]), float(pos[1]), float(pos[2])]
        except Exception:
            return None
    return None


def _view_description(kb: Any, scene_id: str, view_id: str) -> str:
    try:
        tree = kb.scene(scene_id)
    except Exception:
        return ""
    views = (tree or {}).get("views") or {}
    attrs = (views.get(view_id) or {}).get("attributes") or {}
    desc = attrs.get("description")
    return str(desc).strip() if desc is not None else ""


def _distance_score(p1: Optional[List[float]], p2: Optional[List[float]]) -> float:
    if p1 is None or p2 is None:
        return 0.0
    a = np.asarray(p1, dtype=np.float32)
    b = np.asarray(p2, dtype=np.float32)
    if a.shape != (3,) or b.shape != (3,):
        return 0.0
    dist = float(np.linalg.norm(a - b))
    return 1.0 / (1.0 + dist)


def _build_text_embedder(kind: str, config_path: Path, *, binary_dim: int) -> Any:
    from src.retrieval import BinaryRandomEmbedder, build_text_embedder_from_config  # noqa: E402

    if kind == "binary":
        return BinaryRandomEmbedder(dim=binary_dim, threshold=0.3)
    backend = None if kind == "auto" else kind
    return build_text_embedder_from_config(config_path, backend=backend)


def _build_vision_embedder(kind: str, config_path: Path, *, binary_dim: int) -> Any:
    from src.retrieval import BinaryRandomEmbedder, ViTEmbedder  # noqa: E402

    if kind == "binary":
        return BinaryRandomEmbedder(dim=binary_dim, threshold=0.3)
    return ViTEmbedder(config_path=config_path)


def _build_gt_map(gt_csv: Path) -> Dict[str, Dict[str, str]]:
    """按 episode_id 对齐 GT；同一 id 在 CSV 中多次出现时只保留首次出现行。"""
    out: Dict[str, Dict[str, str]] = {}
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = str(row.get("episode_id", "")).strip()
            if not ep or ep in out:
                continue
            out[ep] = {
                "gt_scene_id": str(row.get("gt_scene_id", "")).strip(),
                "gt_start_view_id": str(row.get("gt_start_view_id", "")).strip(),
                "gt_end_view_id": str(row.get("gt_end_view_id", "")).strip(),
                "instruction_text": str(row.get("instruction_text", "")).strip(),
                "start_view_image_path": str(row.get("start_view_image_path", "") or "").strip(),
            }
    return out


def _resolve_start_view_png(
    repo_root: Path,
    start_view_root: Path,
    ep_id: str,
    dataset_file: Path,
) -> Path:
    """
    起点图与 ``raw_data`` / ``raw_data_mask_*`` / ``raw_data_implicit`` 共用一套文件：
    ``data/vln_ce/start_view/r2r/<split>/ep_<id>.png``（自 ``r2r`` 起的短路径，不含 raw_data* 前缀）。
    """
    vln_ce_root = (repo_root / "data" / "vln_ce").resolve()
    ds_parent = dataset_file.expanduser().resolve().parent
    try:
        rel_parts = ds_parent.relative_to(vln_ce_root).parts
    except ValueError as e:
        raise SystemExit(
            f"start_view 路径要求 dataset JSON 位于 data/vln_ce 下: {dataset_file}"
        ) from e
    try:
        i = rel_parts.index("r2r")
    except ValueError as e:
        raise SystemExit(
            f"start_view 共用路径要求在 vln_ce 相对目录中包含 r2r 段: {dataset_file}"
        ) from e
    sub = Path(*rel_parts[i:])
    return (start_view_root / sub / f"ep_{ep_id}.png").resolve()


def _load_robot_image(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    return Image.open(path).convert("RGB")


def _save_kb_view_image(kb: Any, scene_id: str, view_id: str, dest: Path) -> bool:
    p = kb.view_image_path(scene_id, view_id)
    if p is not None and p.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        return True
    img = kb.load_view_image(scene_id, view_id)
    if img is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retriever (Ours) on VLN_CE samples")
    parser.add_argument("--dataset-json", type=Path, required=True, help="VLN_CE episode json file")
    parser.add_argument("--gt-csv", type=Path, default=Path("data/vln_ce/dataset_gt.csv"))
    parser.add_argument("--subset-name", type=str, default="full_instruction", help="Tag for current subset")
    parser.add_argument("--rag4vln-config", type=Path, default=Path("rag4vln/src/config.yaml"))
    parser.add_argument("--kb-root", type=Path, default=Path("rag4vln/data/kb/memory"))
    parser.add_argument("--text-embedder", choices=("auto", "bert", "sbert", "bge", "binary"), default="bge")
    parser.add_argument("--vision-embedder", choices=("vit", "binary"), default="vit")
    parser.add_argument("--binary-dim", type=int, default=64)
    parser.add_argument("--topk1", type=int, default=3)
    parser.add_argument("--topk2", type=int, default=3)
    parser.add_argument(
        "--topk3",
        type=int,
        default=10,
        help="(start,end) 对列表长度及每场景 start/end 候选深度（传给 Retriever.topk3_pairs；默认 10，与旧版单独传 pair_ranking_topk=10 时相当）",
    )
    parser.add_argument(
        "--hit-k",
        type=int,
        default=5,
        help="评测命中率阈值 K（默认 5，对应 Hit@5）",
    )
    parser.add_argument("--max-episodes", type=int, default=0, help="<=0 means all")
    parser.add_argument(
        "--no-export-images",
        action="store_true",
        help="Do not export ins/retriever images to result directory",
    )
    parser.add_argument("--kb-embed-cache", type=Path, default=None)
    parser.add_argument("--rebuild-kb-embed-cache", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=Path("rag4vln/results/retriever_eval"))
    args = parser.parse_args()
    hit_k = max(1, int(args.hit_k))
    hit_k_key = f"hit@{hit_k}"

    repo_root = Path(__file__).resolve().parents[3]
    _setup_sys_path(repo_root)

    dataset_json = args.dataset_json if args.dataset_json.is_absolute() else (repo_root / args.dataset_json).resolve()
    gt_csv = args.gt_csv if args.gt_csv.is_absolute() else (repo_root / args.gt_csv).resolve()
    rag_cfg = args.rag4vln_config if args.rag4vln_config.is_absolute() else (repo_root / args.rag4vln_config).resolve()
    kb_root = args.kb_root if args.kb_root.is_absolute() else (repo_root / args.kb_root).resolve()
    kb_cache = None
    if args.kb_embed_cache is not None:
        kb_cache = args.kb_embed_cache if args.kb_embed_cache.is_absolute() else (repo_root / args.kb_embed_cache).resolve()

    if not dataset_json.is_file():
        raise SystemExit(f"dataset json not found: {dataset_json}")
    if not gt_csv.is_file():
        raise SystemExit(f"gt csv not found: {gt_csv}")
    if args.text_embedder != "binary" or args.vision_embedder == "vit":
        if not rag_cfg.is_file():
            raise SystemExit(f"rag4vln config not found: {rag_cfg}")

    if args.text_embedder == "binary" and args.vision_embedder == "binary":
        binary_dim = max(1, int(args.binary_dim))
    elif args.text_embedder == "binary" or args.vision_embedder == "binary":
        binary_dim = _cfg_embedding_dim(rag_cfg)
    else:
        binary_dim = 768

    from src.kb import KnowledgeBase  # noqa: E402
    from src.retrieval import Retriever  # noqa: E402

    kb = KnowledgeBase(kb_root)
    text_e = _build_text_embedder(args.text_embedder, rag_cfg, binary_dim=binary_dim)
    vision_e = _build_vision_embedder(args.vision_embedder, rag_cfg, binary_dim=binary_dim)
    retriever = Retriever(text_embedder=text_e, vision_embedder=vision_e, caption_config_path=rag_cfg)

    try:
        import PIL.Image  # noqa: F401
    except ImportError:
        raise SystemExit("eval_retriever.py requires Pillow to load start_view images (pip install pillow)")

    gt_map = _build_gt_map(gt_csv)
    start_view_root = (repo_root / "data" / "vln_ce" / "start_view").resolve()
    dataset_obj = json.loads(dataset_json.read_text(encoding="utf-8"))
    episodes = dataset_obj.get("episodes") or []
    if not isinstance(episodes, list) or not episodes:
        raise SystemExit("episodes empty or invalid")
    if int(args.max_episodes) > 0:
        episodes = episodes[: int(args.max_episodes)]

    detail_rows: List[Dict[str, Any]] = []
    scene_hit1: List[float] = []
    scene_hitk: List[float] = []
    start_hit1: List[float] = []
    start_hitk: List[float] = []
    start_mrr: List[float] = []
    end_hit1: List[float] = []
    end_hitk: List[float] = []
    end_mrr: List[float] = []
    end_match_scores: List[float] = []
    end_distance_scores: List[float] = []
    end_semantic_scores: List[float] = []
    end_visual_scores: List[float] = []
    end_semantic_visual_mix_scores: List[float] = []
    end_weighted_distance_scores: List[float] = []
    end_weighted_semantic_visual_scores: List[float] = []

    by_scene: Dict[str, Dict[str, List[float]]] = {}
    by_start_view: Dict[str, Dict[str, List[float]]] = {}
    by_end_view: Dict[str, Dict[str, List[float]]] = {}
    evaled_count = 0
    view_text_emb_cache: Dict[str, Any] = {}
    view_img_emb_cache: Dict[str, Any] = {}

    n_total = len(episodes)

    def _row_ok(e: Dict[str, Any]) -> bool:
        eid = str(e.get("episode_id", "")).strip()
        ins = str(((e.get("instruction") or {}).get("instruction_text") or "")).strip()
        if not eid or not ins:
            return False
        return eid in gt_map

    n_will_eval = sum(1 for e in episodes if _row_ok(e))
    print(
        f"[eval] subset={args.subset_name} episodes={n_total} gt_lookup=episode_id will_eval={n_will_eval}",
        flush=True,
    )

    ts = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.result_dir if args.result_dir.is_absolute() else (repo_root / args.result_dir)
    ).resolve() / f"{args.subset_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_images = not bool(args.no_export_images)
    if export_images:
        dir_ins = out_dir / "ins_start_view"
        dir_rs = out_dir / "retriever_start_view"
        dir_re = out_dir / "retriever_end_view"
        for d in (dir_ins, dir_rs, dir_re):
            d.mkdir(parents=True, exist_ok=True)
        print(f"[eval] image export: {out_dir}", flush=True)
    else:
        print("[eval] image export: disabled", flush=True)

    pbar = tqdm(total=n_total, desc="Retriever Eval", unit="ep") if tqdm is not None else None
    for i, ep in enumerate(episodes):
        ep_id = str(ep.get("episode_id", "")).strip()
        instr = str(((ep.get("instruction") or {}).get("instruction_text") or "")).strip()
        if not ep_id or not instr:
            if pbar is not None:
                pbar.update(1)
            continue
        gt = gt_map.get(ep_id)
        if gt is None:
            if pbar is not None:
                pbar.set_postfix(evaled=f"{evaled_count}/{n_total}", refresh=False)
                pbar.update(1)
            continue

        scene_gt = gt.get("gt_scene_id", "")
        start_gt = gt.get("gt_start_view_id", "")
        end_gt = gt.get("gt_end_view_id", "")

        start_pos = ep.get("start_position")
        if isinstance(start_pos, list) and len(start_pos) == 3:
            robot_position = start_pos
        else:
            robot_position = [0.0, 0.0, 0.0]

        image_path = _resolve_start_view_png(repo_root, start_view_root, ep_id, dataset_json)
        if not image_path.is_file():
            raise SystemExit(
                f"missing start_view image for episode_id={ep_id}: {image_path} "
                f"(run rag4vln/scripts/build_dataset_gt.py or fix start_view_image_path in {gt_csv})"
            )
        robot_image = _load_robot_image(image_path)
        if robot_image is None:
            raise SystemExit(f"failed to load start_view image for episode_id={ep_id}: {image_path}")

        ins_rel = ""
        if export_images:
            ins_rel = f"ins_start_view/ep_{ep_id}.png"
            shutil.copy2(image_path, out_dir / ins_rel)

        plan = retriever.retrieve(
            kb,
            instruction=instr,
            robot_position=robot_position,  # type: ignore[arg-type]
            robot_image=robot_image,
            topk1_scenes=int(args.topk1),
            topk2_zones=int(args.topk2),
            topk3_pairs=int(args.topk3),
            embed_view_images=(args.vision_embedder == "vit"),
            verbose=False,
            timing=False,
            progress=False,
            kb_cache_path=kb_cache,
            force_rebuild_kb_cache=bool(args.rebuild_kb_embed_cache),
        )

        top_scene = plan.get("topk1_scenes") or []
        top_pairs = plan.get("topk3_pairs") or []
        k_export = max(1, int(args.topk3))
        rs_paths: List[str] = []
        re_paths: List[str] = []
        for rank, pair in enumerate(top_pairs[:k_export], start=1):
            if not isinstance(pair, dict):
                continue
            rsid = str(pair.get("scene_id") or "").strip()
            sv = str(pair.get("start_view_id") or "").strip()
            ev = str(pair.get("end_view_id") or "").strip()
            rel_s = f"retriever_start_view/ep_{ep_id}_{rank:02d}.png"
            rel_e = f"retriever_end_view/ep_{ep_id}_{rank:02d}.png"
            if export_images and rsid and sv:
                if _save_kb_view_image(kb, rsid, sv, out_dir / rel_s):
                    rs_paths.append(rel_s)
                else:
                    print(
                        f"[warn] kb view missing start ep={ep_id} rank={rank} scene={rsid} view={sv}",
                        flush=True,
                    )
            if export_images and rsid and ev:
                if _save_kb_view_image(kb, rsid, ev, out_dir / rel_e):
                    re_paths.append(rel_e)
                else:
                    print(
                        f"[warn] kb view missing end ep={ep_id} rank={rank} scene={rsid} view={ev}",
                        flush=True,
                    )

        scene_rank = _rank_in_list(top_scene, scene_gt, key="scene_id")
        start_rank = _rank_in_list(top_pairs, start_gt, key="start_view_id")
        end_rank = _rank_in_list(top_pairs, end_gt, key="end_view_id")
        best_pair = top_pairs[0] if top_pairs and isinstance(top_pairs[0], dict) else {}
        pred_scene = str(best_pair.get("scene_id") or "").strip()
        pred_end_view = str(best_pair.get("end_view_id") or "").strip()

        # 终点匹配分：GT end-view vs 预测 end-view，仅使用语义与视觉相似度
        # score = alpha*semantic + (1-alpha)*visual
        gt_pos = _view_position(kb, scene_gt, end_gt) if scene_gt and end_gt else None
        pred_pos = _view_position(kb, pred_scene, pred_end_view) if pred_scene and pred_end_view else None
        end_distance = _distance_score(gt_pos, pred_pos)

        end_semantic = 0.0
        if scene_gt and end_gt and pred_scene and pred_end_view:
            gt_sem_key = f"{scene_gt}:{end_gt}"
            pred_sem_key = f"{pred_scene}:{pred_end_view}"
            if gt_sem_key not in view_text_emb_cache:
                gt_desc = _view_description(kb, scene_gt, end_gt)
                view_text_emb_cache[gt_sem_key] = text_e.embed(
                    instruction=str(gt_desc), image=None, position=None, rotation=None
                ).get("text_emb")
            if pred_sem_key not in view_text_emb_cache:
                pred_desc = _view_description(kb, pred_scene, pred_end_view)
                view_text_emb_cache[pred_sem_key] = text_e.embed(
                    instruction=str(pred_desc), image=None, position=None, rotation=None
                ).get("text_emb")
            end_semantic = _cosine(view_text_emb_cache[gt_sem_key], view_text_emb_cache[pred_sem_key])

        end_visual = 0.0
        if scene_gt and end_gt and pred_scene and pred_end_view:
            gt_img_key = f"{scene_gt}:{end_gt}"
            pred_img_key = f"{pred_scene}:{pred_end_view}"
            if gt_img_key not in view_img_emb_cache:
                gt_img = kb.load_view_image(scene_gt, end_gt)
                view_img_emb_cache[gt_img_key] = (
                    vision_e.embed(instruction="", image=gt_img, position=None, rotation=None, image_role="kb_view").get("image_feat")
                    if gt_img is not None
                    else None
                )
            if pred_img_key not in view_img_emb_cache:
                pred_img = kb.load_view_image(pred_scene, pred_end_view)
                view_img_emb_cache[pred_img_key] = (
                    vision_e.embed(instruction="", image=pred_img, position=None, rotation=None, image_role="kb_view").get("image_feat")
                    if pred_img is not None
                    else None
                )
            end_visual = _cosine(view_img_emb_cache[gt_img_key], view_img_emb_cache[pred_img_key])

        alpha = float(getattr(retriever, "_alpha", 0.0))
        beta = float(getattr(retriever, "_beta", 0.0))
        end_semantic_visual_mix = float(alpha * end_semantic + (1.0 - alpha) * end_visual)
        end_weighted_distance = 0.0
        end_weighted_semantic_visual = float(end_semantic_visual_mix)
        end_match_score = float(end_semantic_visual_mix)

        s_h1, s_hk = _hit_at(scene_rank, 1), _hit_at(scene_rank, hit_k)
        a_h1, a_hk, a_mrr = _hit_at(start_rank, 1), _hit_at(start_rank, hit_k), _rr(start_rank)
        e_h1, e_hk, e_mrr = _hit_at(end_rank, 1), _hit_at(end_rank, hit_k), _rr(end_rank)

        scene_hit1.append(s_h1)
        scene_hitk.append(s_hk)
        start_hit1.append(a_h1)
        start_hitk.append(a_hk)
        start_mrr.append(a_mrr)
        end_hit1.append(e_h1)
        end_hitk.append(e_hk)
        end_mrr.append(e_mrr)
        end_match_scores.append(end_match_score)
        end_distance_scores.append(float(end_distance))
        end_semantic_scores.append(float(end_semantic))
        end_visual_scores.append(float(end_visual))
        end_semantic_visual_mix_scores.append(end_semantic_visual_mix)
        end_weighted_distance_scores.append(end_weighted_distance)
        end_weighted_semantic_visual_scores.append(end_weighted_semantic_visual)

        sid = _scene_short_id(ep.get("scene_id") or scene_gt)
        by_scene.setdefault(
            sid,
            {
                "scene_hit1": [],
                "scene_hitk": [],
                "start_hit1": [],
                "start_hitk": [],
                "start_mrr": [],
                "end_hit1": [],
                "end_hitk": [],
                "end_mrr": [],
                "end_match_score": [],
            },
        )
        by_scene[sid]["scene_hit1"].append(s_h1)
        by_scene[sid]["scene_hitk"].append(s_hk)
        by_scene[sid]["start_hit1"].append(a_h1)
        by_scene[sid]["start_hitk"].append(a_hk)
        by_scene[sid]["start_mrr"].append(a_mrr)
        by_scene[sid]["end_hit1"].append(e_h1)
        by_scene[sid]["end_hitk"].append(e_hk)
        by_scene[sid]["end_mrr"].append(e_mrr)
        by_scene[sid]["end_match_score"].append(end_match_score)

        if start_gt:
            by_start_view.setdefault(start_gt, {"hit1": [], "hitk": [], "mrr": []})
            by_start_view[start_gt]["hit1"].append(a_h1)
            by_start_view[start_gt]["hitk"].append(a_hk)
            by_start_view[start_gt]["mrr"].append(a_mrr)
        if end_gt:
            by_end_view.setdefault(end_gt, {"hit1": [], "hitk": [], "mrr": [], "score": []})
            by_end_view[end_gt]["hit1"].append(e_h1)
            by_end_view[end_gt]["hitk"].append(e_hk)
            by_end_view[end_gt]["mrr"].append(e_mrr)
            by_end_view[end_gt]["score"].append(end_match_score)

        detail_rows.append(
            {
                "episode_id": ep_id,
                "subset": args.subset_name,
                "scene_id": sid,
                "instruction_text": instr,
                "gt_scene_id": scene_gt,
                "gt_start_view_id": start_gt,
                "gt_end_view_id": end_gt,
                "used_robot_image": True,
                "scene_rank": scene_rank if scene_rank is not None else math.inf,
                "start_view_rank": start_rank if start_rank is not None else math.inf,
                "end_view_rank": end_rank if end_rank is not None else math.inf,
                "scene_hit1": s_h1,
                "scene_hitk": s_hk,
                "start_view_hit1": a_h1,
                "start_view_hitk": a_hk,
                "start_view_mrr": a_mrr,
                "end_view_match_score": end_match_score,
                "end_view_match_components": {
                    "semantic_similarity": end_semantic,
                    "distance_score": end_distance,
                    "visual_similarity": end_visual,
                    "semantic_visual_mix": end_semantic_visual_mix,
                    "weighted_distance": end_weighted_distance,
                    "weighted_semantic_visual": end_weighted_semantic_visual,
                    "alpha": alpha,
                    "beta": beta,
                },
                "image_export": {
                    "ins_start_view": ins_rel,
                    "retriever_start_view": rs_paths,
                    "retriever_end_view": re_paths,
                },
            }
        )
        evaled_count += 1

        if pbar is not None:
            pbar.set_postfix(evaled=f"{evaled_count}/{n_total}", refresh=False)
            pbar.update(1)
        if pbar is None and ((i + 1) % 10 == 0 or (i + 1) == n_total):
            print(f"[eval] processed {i+1}/{n_total}", flush=True)
    if pbar is not None:
        pbar.close()

    overall = {
        "subset": args.subset_name,
        "count": len(detail_rows),
        "scene": {"hit@1": _avg(scene_hit1), hit_k_key: _avg(scene_hitk)},
        "view_start": {"hit@1": _avg(start_hit1), hit_k_key: _avg(start_hitk), "mrr": _avg(start_mrr)},
        "view_end": {
            "hit@1": _avg(end_hit1),
            hit_k_key: _avg(end_hitk),
            "mrr": _avg(end_mrr),
            "distance_score": _avg(end_distance_scores),
            "semantic_similarity": _avg(end_semantic_scores),
            "visual_similarity": _avg(end_visual_scores),
            "semantic_visual_mix": _avg(end_semantic_visual_mix_scores),
            "weighted_distance": _avg(end_weighted_distance_scores),
            "weighted_semantic_visual": _avg(end_weighted_semantic_visual_scores),
            "match_score": _avg(end_match_scores),
        },
        "settings": {
            "dataset_json": str(dataset_json),
            "gt_csv": str(gt_csv),
            "topk1": int(args.topk1),
            "topk2": int(args.topk2),
            "topk3": int(args.topk3),
            "hit_k": hit_k,
            "text_embedder": args.text_embedder,
            "vision_embedder": args.vision_embedder,
            "kb_root": str(kb_root),
            "require_start_view_image": True,
            "gt_lookup": "episode_id",
            "export_images": export_images,
            "image_export_subdirs": (
                ["ins_start_view", "retriever_start_view", "retriever_end_view"] if export_images else []
            ),
        },
    }

    by_scene_out = {
        sid: {
            "count": len(vals["scene_hit1"]),
            "scene_hit@1": _avg(vals["scene_hit1"]),
            f"scene_{hit_k_key}": _avg(vals["scene_hitk"]),
            "start_view": {
                "hit@1": _avg(vals["start_hit1"]),
                hit_k_key: _avg(vals["start_hitk"]),
                "mrr": _avg(vals["start_mrr"]),
            },
            "end_view": {
                "hit@1": _avg(vals["end_hit1"]),
                hit_k_key: _avg(vals["end_hitk"]),
                "mrr": _avg(vals["end_mrr"]),
                "match_score": _avg(vals["end_match_score"]),
            },
        }
        for sid, vals in by_scene.items()
    }
    def _view_bucket_out(bucket: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Any]]:
        return {
            vid: {
                "count": len(vals.get("hit1", vals.get("score", []))),
                **(
                    {
                        "hit@1": _avg(vals["hit1"]),
                        hit_k_key: _avg(vals["hitk"]),
                        "mrr": _avg(vals["mrr"]),
                    }
                    if "hit1" in vals
                    else {"match_score": _avg(vals.get("score", []))}
                ),
            }
            for vid, vals in bucket.items()
        }

    by_start_view_out = _view_bucket_out(by_start_view)
    by_end_view_out = _view_bucket_out(by_end_view)

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "overall": overall,
                "by_scene": by_scene_out,
                "by_start_view": by_start_view_out,
                "by_end_view": by_end_view_out,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in detail_rows),
        encoding="utf-8",
    )
    (out_dir / "result.txt").write_text(
        "\n".join(
            [
                f"subset={overall['subset']}",
                f"count={overall['count']}",
                f"scene_hit@1={overall['scene']['hit@1']:.6f}",
                f"scene_{hit_k_key}={overall['scene'][hit_k_key]:.6f}",
                f"view_start_hit@1={overall['view_start']['hit@1']:.6f}",
                f"view_start_{hit_k_key}={overall['view_start'][hit_k_key]:.6f}",
                f"view_start_mrr={overall['view_start']['mrr']:.6f}",
                f"view_end_hit@1={overall['view_end']['hit@1']:.6f}",
                f"view_end_{hit_k_key}={overall['view_end'][hit_k_key]:.6f}",
                f"view_end_mrr={overall['view_end']['mrr']:.6f}",
                f"view_end_distance_score={overall['view_end']['distance_score']:.6f}",
                f"view_end_semantic_similarity={overall['view_end']['semantic_similarity']:.6f}",
                f"view_end_visual_similarity={overall['view_end']['visual_similarity']:.6f}",
                f"view_end_semantic_visual_mix={overall['view_end']['semantic_visual_mix']:.6f}",
                f"view_end_weighted_distance={overall['view_end']['weighted_distance']:.6f}",
                f"view_end_weighted_semantic_visual={overall['view_end']['weighted_semantic_visual']:.6f}",
                f"view_end_match_score={overall['view_end']['match_score']:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[eval] done. output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
