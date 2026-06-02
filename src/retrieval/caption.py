# -*- coding: utf-8 -*-
"""
Robot observation image → multimodal caption (see `retrieval.caption` in `src/config.yaml`).
Called from `Retriever.retrieve`, not inside text `Embedder` / `ViTEmbedder`.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..config_io import DEFAULT_CONFIG_PATH, load_retrieval_config


def _resolve_api_key(key_spec: str) -> str:
    """Resolve API key from environment variable name only (see config ``api_key_env``)."""
    if not key_spec:
        return ""
    name = key_spec.strip()
    return os.environ.get(name, "")


def _resize_max_side(img: Any, max_side: int) -> Any:
    from PIL import Image as PILImage

    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), PILImage.BICUBIC)


def _jpeg_data_url(pil: Any, cap_cfg: Dict[str, Any]) -> str:
    vis = cap_cfg.get("vision") or {}
    max_side = int(vis.get("image_max_side", 768))
    q = int(vis.get("jpeg_quality", 85))
    im = _resize_max_side(pil, max_side)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class DashScopeRobotCaptioner:
    """Alibaba DashScope OpenAI-compatible API; default qwen-vl-max."""

    def __init__(self, caption_cfg: Dict[str, Any]):
        self._cfg = caption_cfg
        self._client: Optional[Any] = None
        self._api_key_env = str(caption_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
        self._client_inited = False

    def _lazy_init_client(self) -> None:
        if self._client_inited:
            return
        self._client_inited = True
        if not self._cfg.get("enabled", True):
            raise RuntimeError("retrieval.caption.enabled is false")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            raise RuntimeError("openai package is not installed for captioning") from e
        base = str(self._cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        api_key = _resolve_api_key(self._api_key_env)
        if not api_key:
            raise RuntimeError(f"missing caption api key env: {self._api_key_env}")
        self._client = OpenAI(api_key=api_key, base_url=base)

    def caption(self, image: Any) -> str:
        """Generate an English description for the robot RGB observation."""
        from .image_utils import pil_from_any

        self._lazy_init_client()
        if self._client is None:
            raise RuntimeError("caption client is not initialized")

        pil = pil_from_any(image)
        model = str(self._cfg.get("model", "qwen-vl-max"))
        timeout = float(self._cfg.get("timeout_sec", 120))
        max_tokens = int(self._cfg.get("max_tokens", 512))
        temperature = float(self._cfg.get("temperature", 0.2))
        sys_p = str(self._cfg.get("system_prompt", "")).strip()
        usr_p = str(self._cfg.get("user_prompt", "")).strip()
        vis_cfg = self._cfg.get("vision") or {}
        use_vision = bool(vis_cfg.get("enabled", True))

        if use_vision:
            data_url = _jpeg_data_url(pil, self._cfg)
            messages = [
                {"role": "system", "content": sys_p},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": usr_p},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ]
        else:
            suffix = str(self._cfg.get("user_prompt_no_image_suffix", "")).strip()
            user_text = usr_p + (f"\n{suffix}" if suffix else "")
            messages = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_text},
            ]
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("empty caption content")
        return text


def build_robot_captioner(config_path: Optional[Path] = None) -> DashScopeRobotCaptioner:
    """Load `caption` from the `retrieval` section of unified `config.yaml` and build a captioner."""
    p = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    if not p.is_file():
        return DashScopeRobotCaptioner({})
    r = load_retrieval_config(p)
    return DashScopeRobotCaptioner(r.get("caption") or {})
