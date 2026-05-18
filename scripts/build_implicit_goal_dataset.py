#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an implicit goal dataset from VLN-CE raw_data.

This script follows dataset_generation-style I/O:
- recursively read .json / .json.gz files under input root
- rewrite episodes[*].instruction.instruction_text
- set episodes[*].instruction.instruction_tokens = null
- by default does **not** write ``original_instruction_text`` (use ``--save-original`` to keep it)

Generation style:
- use an LLM to rewrite each navigation instruction into a concise
  goal-only implicit instruction in English.

Preview a few episodes without rewriting the whole corpus:

  python rag4vln/scripts/build_implicit_goal_dataset.py \\
    --input-root data/vln_ce/raw_data \\
    --output-root data/vln_ce/raw_data_implicit_goal_preview10 \\
    --max-episodes 10
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ORIGINAL_INSTRUCTION_TEXT_KEY = "original_instruction_text"

DEFAULT_SYSTEM_PROMPT = """You are an instruction abstracter for indoor navigation tasks.
Rewrite each original navigation instruction into an implicit, incomplete goal-only instruction.
Requirements:
1) Output exactly one natural-language sentence in English.
2) Keep only the user's intent; remove route details and intermediate landmarks.
3) No explanation, no bullet list, no quotation marks.
4) Keep it concise (about 4-12 words).
5) Prefer functional association from destination semantics
   (e.g., "arrive at the bar counter" -> "I want a drink").
"""

DEFAULT_USER_PROMPT_TEMPLATE = """Original navigation instruction:
{instruction}

Output the rewritten implicit goal-only instruction in English."""

# rag4vln/（含 src/），与其它 scripts 一致：加入 sys.path 后用 from src... 导入
RAG4VLN_ROOT = Path(__file__).resolve().parents[1]


def _setup_sys_path() -> None:
    if str(RAG4VLN_ROOT) not in sys.path:
        sys.path.insert(0, str(RAG4VLN_ROOT))


_setup_sys_path()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_api_key(key_spec: str) -> str:
    if not key_spec:
        return ""
    return os.environ.get(key_spec.strip(), "")


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_json(path: Path) -> Dict[str, Any]:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _iter_dataset_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    yield from sorted(root.rglob("*.json"))
    for p in sorted(root.rglob("*.gz")):
        if p.name.endswith(".json.gz"):
            yield p


def _load_llm_cfg(config_path: Path) -> Dict[str, Any]:
    from src.config_io import load_augment_config  # noqa: E402

    aug = load_augment_config(config_path)
    llm = aug.get("llm")
    return dict(llm) if isinstance(llm, dict) else {}


class ImplicitGoalLLM:
    def __init__(
        self,
        *,
        config_path: Path,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        self._cfg = _load_llm_cfg(config_path)
        self._system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self._user_prompt_template = user_prompt_template.strip() or DEFAULT_USER_PROMPT_TEMPLATE
        self._client = None
        self._client_inited = False
        self._client_unavailable_reason: Optional[str] = None

    def _lazy_init_client(self) -> None:
        if self._client_inited:
            return
        self._client_inited = True
        if not self._cfg.get("enabled", True):
            self._client_unavailable_reason = "augment.llm.enabled is false in config"
            return
        try:
            from openai import OpenAI
        except ModuleNotFoundError as e:
            self._client_unavailable_reason = "openai package is not installed"
            raise RuntimeError(self._client_unavailable_reason) from e
        base = str(self._cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        key = _resolve_api_key(str(self._cfg.get("api_key_env", "DASHSCOPE_API_KEY")))
        if not key:
            self._client_unavailable_reason = (
                "could not resolve API key from augment.llm.api_key_env (set env var or valid key)"
            )
            return
        self._client = OpenAI(api_key=key, base_url=base)

    def generate(self, instruction: str) -> str:
        self._lazy_init_client()
        if not self._cfg.get("enabled", True):
            raise RuntimeError(self._client_unavailable_reason or "augment.llm is disabled in config")
        if self._client is None:
            raise RuntimeError(self._client_unavailable_reason or "LLM client is not initialized")

        model = str(self._cfg.get("model", "qwen-plus"))
        timeout = float(self._cfg.get("timeout_sec", 120))
        max_tokens = int(self._cfg.get("max_tokens", 128))
        temperature = float(self._cfg.get("temperature", 0.2))
        user_prompt = self._user_prompt_template.format(instruction=(instruction or "").strip())

        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}") from e

        raw = (resp.choices[0].message.content or "").strip()
        text = _strip_code_fence(raw)
        if not text:
            raise RuntimeError("LLM returned empty content")
        return text


def _process_episodes(
    data: Dict[str, Any],
    llm: ImplicitGoalLLM,
    budget: Optional[List[int]] = None,
    *,
    save_original: bool = False,
) -> tuple[int, int, int]:
    """
    Transform episodes with optional global LLM budget.
    budget: if set, must be a one-element list budget[0] = remaining LLM calls;
            decremented per transformed episode; episodes skipped when 0.
    Returns (n_transformed, n_skip_invalid, n_left_unchanged_valid).
    """
    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        return 0, 0, 0

    n_ok = 0
    n_skip = 0
    n_unchanged = 0
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        instr = ep.get("instruction")
        if not isinstance(instr, dict):
            n_skip += 1
            continue
        text = instr.get("instruction_text")
        if not isinstance(text, str):
            n_skip += 1
            continue
        if budget is not None:
            if budget[0] <= 0:
                n_unchanged += 1
                continue
            budget[0] -= 1
        if save_original:
            instr[ORIGINAL_INSTRUCTION_TEXT_KEY] = text
        else:
            instr.pop(ORIGINAL_INSTRUCTION_TEXT_KEY, None)
        instr["instruction_text"] = llm.generate(text)
        instr["instruction_tokens"] = None
        n_ok += 1
    return n_ok, n_skip, n_unchanged


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build implicit goal-only dataset with LLM")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/vln_ce/raw_data"),
        help="Input root (relative to repo root or absolute path), recursively reads .json/.json.gz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/vln_ce/raw_data_implicit"),
        help="Output root (relative to repo root or absolute path), preserving relative structure",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="If > 0, only LLM-rewrite this many episodes globally (in file order); "
        "then stop without writing any further input files",
    )
    parser.add_argument(
        "--rag4vln-config",
        type=Path,
        default=Path("rag4vln/src/config.yaml"),
        help="Config path for augment.llm (base_url/model/api_key_env/etc.)",
    )
    parser.add_argument("--system-prompt", type=str, default="", help="Optional override for system prompt")
    parser.add_argument(
        "--user-prompt-template",
        type=str,
        default="",
        help="Optional override for user prompt template; must contain {instruction}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print files that would be processed")
    parser.add_argument(
        "--save-original",
        action="store_true",
        help="Also write original_instruction_text before overwriting instruction_text",
    )
    args = parser.parse_args(argv)

    repo = _repo_root()
    in_root = args.input_root if args.input_root.is_absolute() else (repo / args.input_root).resolve()
    out_root = args.output_root if args.output_root.is_absolute() else (repo / args.output_root).resolve()
    cfg_path = args.rag4vln_config if args.rag4vln_config.is_absolute() else (repo / args.rag4vln_config).resolve()

    user_prompt_template = args.user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE
    if "{instruction}" not in user_prompt_template:
        raise SystemExit("--user-prompt-template must include {instruction}")
    if not in_root.is_dir():
        raise SystemExit(f"input-root is not a directory: {in_root}")

    llm = ImplicitGoalLLM(
        config_path=cfg_path,
        system_prompt=args.system_prompt or DEFAULT_SYSTEM_PROMPT,
        user_prompt_template=user_prompt_template,
    )

    files = list(_iter_dataset_files(in_root))
    if not files:
        print(f"[build_implicit_goal_dataset] no json/json.gz found: {in_root}", file=sys.stderr)
        return

    max_ep = int(args.max_episodes)
    budget: Optional[List[int]] = [max_ep] if max_ep > 0 else None

    print(f"[build_implicit_goal_dataset] input={in_root}\n[build_implicit_goal_dataset] output={out_root}", flush=True)
    print(f"[build_implicit_goal_dataset] config={cfg_path}", flush=True)
    if budget is not None:
        print(f"[build_implicit_goal_dataset] max_episodes={budget[0]} (then stop; remaining files skipped)", flush=True)

    total_ep = 0
    total_skip = 0
    total_unchanged = 0
    for src in files:
        try:
            rel = src.relative_to(in_root)
        except ValueError:
            continue
        dst = out_root / rel
        if args.dry_run:
            print(f"  would process: {rel}")
            continue

        if budget is not None and budget[0] <= 0:
            break

        data = _load_json(src)
        n_ok, n_skip, n_left = _process_episodes(
            data,
            llm,
            budget,
            save_original=bool(args.save_original),
        )
        total_ep += n_ok
        total_skip += n_skip
        total_unchanged += n_left
        _save_json(dst, data)
        print(
            f"  wrote {rel} episodes_transformed={n_ok} skipped={n_skip} left_unchanged_in_file={n_left}",
            flush=True,
        )

    if not args.dry_run:
        print(
            f"[build_implicit_goal_dataset] done episodes_transformed={total_ep} "
            f"skipped={total_skip} left_unchanged_in_partial_files={total_unchanged}",
            flush=True,
        )


if __name__ == "__main__":
    main()
