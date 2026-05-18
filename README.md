# rag4vln

**Languages:** **English** · [简体中文](README.zh-CN.md)

A **RAG plugin** for vision-and-language navigation (VLN): before downstream navigation evaluation, it retrieves scene / zone / view evidence from a memory knowledge base (KB) using the user intent and the robot’s starting observation, then optionally runs pluggable instruction augmenters to produce more executable navigation instructions. Core retrieval and augmentation are **decoupled** from any specific VLN backbone; glue scripts under `scripts/eval/` connect to InternNav, StreamVLN, and similar frameworks.

**Working directory:** Unless noted otherwise, run all commands from the **InternNav repository root** (the directory that contains both `rag4vln/` and `internnav/`).

---

## 1. Introduction

### 1.1 Overview

rag4vln inserts two stages into a standard VLN pipeline:

1. **Retrieval:** `Retriever` matches scenes, zones, and start/end views in the KB and returns path evidence on the zone graph.
2. **Instruction augmentation:** `InstructionAugmenter` fuses incomplete or implicit instructions with retrieval evidence into full, executable `instruction_text`.

Downstream Habitat evaluation only needs the **augmented** dataset. This repo does not train navigation policies—it handles **retrieval + instruction rewriting** only.

**Conventions**

- KB hierarchy: `scene` → `zone` → `view`, stored under `rag4vln/data/kb/<name>/scenes/*.json`.
- Multiple viewpoints are **flattened** into distinct `view` entries at build time (unique IDs).
- Retrieval paths use the **zone graph**, not a view graph.
- `instances` (object-level details inside views) are not part of the core schema.

**Pipeline (downstream-agnostic part)**

```
User instruction + start_view image → Retriever → structured evidence → Augmenter → augmented instruction
                                                                                      ↓
                                                                            temp dataset (.json.gz)
                                                                                      ↓
                                                                            downstream VLN Habitat eval
```

### 1.2 Repository layout

```
rag4vln/
├── src/                          # Core Python package (imported by demo / eval)
│   ├── config.yaml               # Unified config: basic / retrieval / augment
│   ├── config_io.py              # Config loader (legacy flat YAML compatible)
│   ├── kb/                       # KB build & load
│   ├── retrieval/                # Embeddings, caption, retrieval scoring
│   ├── augment/                  # Pluggable instruction augmenters
│   └── utils/                    # Habitat rendering, I/O helpers
├── data/
│   ├── kb/                       # Built KB (scenes, imgs, manifest)
│   ├── memory/                   # Source data for KB build (annotations, connectivity)
│   └── test_materials/           # Sample images for demos
├── scripts/
│   ├── demo/                     # Quick retrieval / augmentation trials
│   ├── eval/                     # Retrieval metrics & augment + downstream VLN eval
│   │   └── configs/              # InternNav eval / Habitat dataset configs
│   ├── build_memory_kb.py        # KB build entry point
│   └── build_dataset_gt.py       # GT table & start_view images for retrieval eval
├── results/                      # Run outputs (recommended gitignore)
├── README.md
├── README.zh-CN.md
└── instruction.md                # Command cheat sheet
```

### 1.3 Code map

| Path | Role |
|------|------|
| `src/kb/kb.py` | `KnowledgeBase`: load scene JSON |
| `src/kb/kb_build.py` | Build KB from memory data; optional view rendering |
| `src/retrieval/retriever.py` | `Retriever.retrieve`: hierarchical retrieval & zone paths |
| `src/retrieval/embedder.py` | Text/image embeddings (BERT, SBERT, BGE, ViT, binary) |
| `src/retrieval/caption.py` | VLM caption for robot observations (e.g. DashScope) |
| `src/augment/instruction_augmenter.py` | Augmenter base class |
| `src/augment/*_augmenter.py` | Concrete augmentation strategies |
| `scripts/demo/test_retriever_demo.py` | Retrieval-only demo |
| `scripts/demo/test_augmenter_demo.py` | Retrieval + augmentation demo |
| `scripts/eval/eval_retriever.py` | Offline retrieval metrics (Hit@K, MRR) |
| `scripts/eval/eval_rag4vln_vln_augmented.py` | Augment then **InternNav** Habitat eval |

**Scene JSON (simplified)**

- `scene.attributes`: `description`, `zone_ids`, `zone_graph.adjacency`, `view_ids`
- `zones[zone_id].attributes`: `description`, `scene_id`, `adjacent_zone_ids`, `view_ids`
- `views[view_id].attributes`: `description`, `position`, `rotation`, `zone_id`, `img`, `included`

**`Retriever.retrieve` outputs**

- `topk1_scenes`, `topk2_zones`, `topk3_pairs`
- Common fields in `topk3_pairs`: `start_zone_id`, `start_view_id`, `end_zone_id`, `end_view_id`, `scores`, `path_zone_ids`

**Instruction augmenters (`--augmenter`)**

| Name | Description |
|------|-------------|
| `llm_direct` | Single-shot LLM rewrite |
| `template_path` | LLM slot filling + local template |
| `semantic_pathplanning` | Three-stage CoT + final VLN-style instruction (eval default) |
| `r_only` | Concatenate retrieval evidence with original text only (no LLM baseline) |

---

## 2. Installation

We recommend installing **InternNav Habitat** and **rag4vln** dependencies in the **same Conda environment** so you can run end-to-end Habitat evaluation (§5) directly. The examples below use env name `inter_hab` (as in eval script hints); you may rename it.

### 2.1 InternNav Habitat environment

VLN-CE evaluation requires **InternNav + bundled Habitat-Lab**. Set up the simulator at the **InternNav repo root** (with `internnav/`, `habitat-lab/`, and `rag4vln/`).

**1. Clone the repo**

```bash
git clone https://github.com/InternRobotics/InternNav.git --recursive
cd InternNav
# If rag4vln is separate, place it at: InternNav/rag4vln/
```

**2. Create a Conda environment**

```bash
conda create -n inter_hab python=3.10 -y
conda activate inter_hab
```

Follow the Python version in the [InternNav installation guide](https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html) (typically 3.9–3.10).

**3. Install Habitat-Sim**

```bash
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

Use a version compatible with the repo’s `habitat-lab`; if the official docs pin `habitat-sim==x.y`, follow that.

**4. Install vendored Habitat-Lab / Baselines**

```bash
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
```

**5. Install InternNav (Habitat + eval model extras)**

```bash
pip install -e ".[habitat,internvla_n1]"
```

Notes:

- `[habitat]`: VLN-CE simulation and `internnav.habitat_extensions` (see `requirements/habitat_requirements.txt`).
- `[internvla_n1]`: Dependencies for the default dual-system config in `eval_rag4vln_vln_augmented.py` (see `requirements/internvla_n1.txt`). For retrieval-only eval without InternVLA, `[habitat]` alone is enough.

See [Quick Start — Installation](https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html) for GPU/CUDA, checkpoints, and optional components.

**6. Sanity check (optional)**

```bash
python -c "import habitat; import internnav; print('habitat + internnav OK')"
```

### 2.2 rag4vln dependencies

rag4vln has **no standalone `setup.py`**. Install extra Python packages in the active `inter_hab` (or your InternNav) environment; code is loaded via `sys.path` on the `rag4vln/` directory.

**1. Common packages for retrieval and evaluation**

```bash
conda activate inter_hab
pip install \
  numpy \
  pyyaml \
  tqdm \
  pillow \
  torch \
  transformers \
  sentence-transformers \
  openai
```

| Use case | Main dependencies |
|----------|-------------------|
| Retrieval embedders `bge` / `vit` | `torch`, `transformers`, `sentence-transformers` |
| Retrieval `binary` (quick demo) | `numpy` only |
| VLM caption + LLM augmentation | `openai` (OpenAI-compatible APIs e.g. DashScope), `pillow` |
| KB build / `build_dataset_gt` rendering | Above + **Habitat-Sim** from §2.1 |
| Scripts & config | `pyyaml`, `tqdm` |

First run with `bge` / `vit` downloads weights from Hugging Face; ensure network access or a mirror.

**2. API configuration (caption & augmentation)**

Edit `api_key_env` / `base_url` / `model` under `retrieval.caption` and `augment.*` in `rag4vln/src/config.yaml`. Prefer environment variables for secrets:

```bash
export DASHSCOPE_API_KEY="your-key"   # example: Alibaba DashScope
```

For offline retrieval trials, use demo flags `--no-robot-image`, eval `--augmenter r_only`, or `--text-embedder binary --vision-embedder binary`.

**3. Sanity check (optional)**

From the InternNav repo root:

```bash
python rag4vln/scripts/demo/test_retriever_demo.py \
  --text-embedder binary --vision-embedder binary --no-save-result
```

If the KB from §3 is ready and you want semantic embeddings:

```bash
python rag4vln/scripts/demo/test_retriever_demo.py \
  --text-embedder bge --vision-embedder vit --no-save-result
```

**4. Long jobs with `conda run`**

For live logs during long evaluation:

```bash
conda run --no-capture-output -n inter_hab python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py ...
```

---

## 3. Data preparation

To run the official evaluation pipeline, prepare three data categories (paths relative to the **InternNav repo root**, alongside `rag4vln/` and `internnav/`).

```
<repo_root>/
├── data/
│   ├── scene_data/mp3d_ce/          # ① Scene meshes (InternNav)
│   └── vln_ce/                      # ① Original R2R + ③ Eval instruction variants
│       ├── raw_data/
│       ├── raw_data_mask_1/
│       ├── raw_data_implicit/       # or raw_data_mask_0.5 per bundle layout
│       ├── start_view/              # Start images for retrieval eval (script-generated)
│       └── dataset_gt.csv
└── rag4vln/
    └── data/
        ├── memory/                  # ② NavRAG memory (from bundle)
        └── kb/memory/               # Built from memory (below)
```

### 3.1 Scenes and original instructions (InternNav)

Prepare VLN-CE assets from **[InternNav](https://github.com/InternRobotics/InternNav)** and place them under `data/`:

| Content | Path | Notes |
|---------|------|-------|
| MP3D scenes (VLN-CE rearranged) | `data/scene_data/mp3d_ce/mp3d/<scene_id>/` | Required for Habitat sim and KB view rendering (`--scene-root`) |
| R2R episodes (full instructions) | `data/vln_ce/raw_data/r2r/<split>/<split>.json.gz` | Official VLN-CE R2R; splits e.g. `val_seen`, `val_unseen` |

Follow InternNav docs for **VLN-CE / scene data**, for example:

- Scenes: MP3D-CE from InternData-N1 / Scene-N1 (e.g. `mp3d_pe` → `data/scene_data/`)
- Instructions: Habitat official R2R VLN-CE zip or InternNav-converted `json.gz` layout

See [InternNav quick start](https://internrobotics.github.io/user_guide/internnav/quick_start/index.html) for exact download links.

### 3.2 Memory (NavRAG)

The retrieval KB is built from **NavRAG** scene / zone / view annotations and connectivity graphs. Extract to:

```text
rag4vln/data/memory/instruction_generator/
```

After extraction, you should have at least (matching `scripts/build_memory_kb.py` defaults):

| File / directory | Purpose |
|------------------|---------|
| `mp3d_view_annotation.json` | View descriptions & poses |
| `mp3d_zone_annotation.json` | Zone segmentation |
| `mp3d_house_annotation.json` | House-level annotations (optional) |
| `connectivity_mp3d/` | Per-scan MP3D connectivity JSON |

**Standalone download:** Follow [NavRAG](https://github.com/MrZihan/NavRAG) and place the `instruction_generator` tree under `rag4vln/data/memory/`.

**Recommended:** Use the **§3.4 data bundle** (includes memory; no separate NavRAG download).

Build the KB (Habitat installed, scenes from §3.1 ready; add `--render-view-images` to bake view images):

```bash
python rag4vln/scripts/build_memory_kb.py \
  --memory-dir rag4vln/data/memory/instruction_generator \
  --output-dir rag4vln/data/kb/memory \
  --render-view-images \
  --scene-root data/scene_data/mp3d_ce/mp3d
```

Outputs: `rag4vln/data/kb/memory/scenes/*.json`, and optionally `imgs/<scene_id>/<view_id>.png`.

### 3.3 Evaluation instruction sets

Instruction variants for **retrieval eval** (`eval_retriever.py`) and **augment + Habitat eval** (`eval_rag4vln_vln_augmented.py`) live at the repo root:

```text
data/vln_ce/
├── raw_data/              # Full instructions (may overlap §3.1)
├── raw_data_mask_1/       # Masked / incomplete instructions (common in examples)
├── raw_data_implicit/     # Implicit-goal instructions (names may vary in bundle)
└── dataset_gt.csv         # Optional: from bundle or generated by script
```

Same layout as `raw_data`, e.g.:

`data/vln_ce/raw_data_mask_1/r2r/val_seen/val_seen.json.gz`

**Extra step for retrieval eval** — after KB and scenes are ready, build GT and shared start views:

```bash
python rag4vln/scripts/build_dataset_gt.py
# Implicit set: python rag4vln/scripts/build_dataset_gt.py --vln-subdir raw_data_implicit
```

Defaults: `data/vln_ce/dataset_gt.csv`, `data/vln_ce/start_view/r2r/<split>/ep_<episode_id>.png`.

**Standalone:** Download only the `vln_ce` subtree from project Releases if needed; **the §3.4 bundle is recommended**.

### 3.4 Data bundle (memory + eval instructions)

We provide a **rag4vln data bundle** with **§3.2 memory** and **§3.3 evaluation instruction sets** to avoid scattered downloads.

| Step | Action |
|------|--------|
| 1. Download | **(Link TBD — see GitHub Releases or README updates)** |
| 2. Extract memory | Map bundle `memory/instruction_generator/` → `rag4vln/data/memory/instruction_generator/` |
| 3. Extract eval data | Map bundle `vln_ce/` → repo root `data/vln_ce/` (do not overwrite your own `raw_data` unless intended) |
| 4. Build KB | Run `build_memory_kb.py` from §3.2 |
| 5. Build GT (if `dataset_gt.csv` missing) | Run `build_dataset_gt.py` from §3.3 |

The bundle does **not** include MP3D scene meshes or the full official R2R package; obtain those via **§3.1** from InternNav / Habitat.

---

## 4. Quick start

This section validates **retrieval and instruction augmentation** without a full Habitat navigation run. Pick embedders and augmenter via `--text-embedder`, `--vision-embedder`, and `--augmenter`. For semantic quality use `bge` + `vit` (models and APIs in `src/config.yaml`).

### 4.1 Retrieval only

```bash
python rag4vln/scripts/demo/test_retriever_demo.py --text-embedder bge --vision-embedder vit
```

Lightweight trial (random binary embeddings, no large models):

```bash
python rag4vln/scripts/demo/test_retriever_demo.py --text-embedder binary --vision-embedder binary
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--text-embedder` | `auto` \| `bert` \| `sbert` \| `bge` \| `binary` | `auto` |
| `--vision-embedder` | `vit` \| `binary` | `vit` |
| `--config` | Unified YAML (`retrieval` section) | `rag4vln/src/config.yaml` |
| `--kb-root` | KB root | `rag4vln/data/kb/memory` |
| `--instruction` | Query text | Built-in English example |
| `--binary-dim` | `binary` embedding dimension | `64` |
| `--robot-image` | Robot observation (caption + retrieval) | `rag4vln/data/test_materials/test.png` |
| `--no-robot-image` | Skip image & VLM caption | off |
| `--result-dir` | Output directory | `rag4vln/results` |
| `--no-save-result` | Do not write `plan.json` etc. | off |

### 4.2 Retrieval + instruction augmentation

```bash
python rag4vln/scripts/demo/test_augmenter_demo.py \
  --augmenter semantic_pathplanning \
  --text-embedder bge \
  --vision-embedder vit
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--augmenter` | `llm_direct` \| `template_path` \| `semantic_pathplanning` | `llm_direct` |
| `--text-embedder` | Same as §4.1 | `auto` |
| `--vision-embedder` | Same as §4.1 | `vit` |
| `--config` | Unified YAML (`retrieval` + `augment`) | `rag4vln/src/config.yaml` |
| `--robot-image` | Robot image | `rag4vln/data/test_materials/test.png` |
| `--instruction` | User intent | Built-in English example |
| `--episode-id` | Load start image & instruction from GT CSV | none |
| `--gt-csv` | GT table with `start_view_image_path` | `data/vln_ce/dataset_gt.csv` |
| `--binary-dim` | `binary` embedding dimension | `64` |
| `--result-dir` | Output directory | `rag4vln/results` |
| `--no-save-result` | Skip `plan.json` / `evidence.json` / `augmentation.json` | off |

On success, check `rag4vln/results/<timestamp>/` for `plan.json`, `evidence.json`, and `augmentation.json`.

### 4.3 Other debug scripts (optional)

| Script | Purpose |
|--------|---------|
| `scripts/demo/visualize_kb_views.py` | Export KB view PNGs and sidecar JSON for a scene |
| `scripts/demo/visualize_connectivity_mp3d.py` | Summarize MP3D connectivity JSON |

---

## 5. Evaluation

### 5.1 Retrieval evaluation

Script: `rag4vln/scripts/eval/eval_retriever.py`

Measures retrieval on VLN-CE subsets with ground truth. Main metrics:

- **Scene:** `Hit@1` / `Hit@K` (default `K=5`)
- **View:** `Hit@1` / `Hit@K` / `MRR` (start and end separately; MRR is 0 if GT is outside `topk3_pairs`)

For `Hit@K`, use **`--topk3 >= K`** (defaults: `--topk3 10`, `--hit-k 5`).

**Dataset variants** (`--dataset-json`):

- Full: `data/vln_ce/raw_data/r2r/...`
- Masked: `data/vln_ce/raw_data_mask_1/r2r/...`
- Implicit: `data/vln_ce/raw_data_implicit/r2r/...` (or `raw_data_mask_0.5` depending on layout)

GT table default: `data/vln_ce/dataset_gt.csv` (join on **`episode_id`**; first row wins on duplicates). Start images: `data/vln_ce/start_view/r2r/<split>/ep_<episode_id>.png` (generate with `scripts/build_dataset_gt.py`).

**Example command**

```bash
python rag4vln/scripts/eval/eval_retriever.py \
  --dataset-json data/vln_ce/raw_data_mask_1/r2r/val_seen/val_seen.json \
  --gt-csv data/vln_ce/dataset_gt.csv \
  --text-embedder bge --vision-embedder vit \
  --topk1 3 --topk2 3 --topk3 10 --hit-k 5 \
  --max-episodes 150 --no-export-images \
  --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt \
  --subset-name val_seen_mask_1
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-json` | VLN-CE episode JSON (required) | — |
| `--gt-csv` | GT alignment table | `data/vln_ce/dataset_gt.csv` |
| `--subset-name` | Subset label in outputs | `full_instruction` |
| `--rag4vln-config` | Retrieval YAML | `rag4vln/src/config.yaml` |
| `--kb-root` | KB root | `rag4vln/data/kb/memory` |
| `--text-embedder` | `auto` \| `bert` \| `sbert` \| `bge` \| `binary` | `bge` |
| `--vision-embedder` | `vit` \| `binary` | `vit` |
| `--topk1` / `--topk2` / `--topk3` | Scene / zone / pair list depth | `3` / `3` / `10` |
| `--hit-k` | K for `Hit@K` | `5` |
| `--max-episodes` | First N if `>0`; all if `<=0` | `0` |
| `--no-export-images` | Skip comparison image export | off |
| `--kb-embed-cache` | KB embedding cache `.pt` | none |
| `--result-dir` | Output root | `rag4vln/results/retriever_eval` |

Main outputs: `metrics.json`, `details.jsonl`, `result.txt`; without `--no-export-images`, also `ins_start_view/`, `retriever_start_view/`, `retriever_end_view/`.

### 5.2 Instruction augmentation + downstream VLN eval (InternNav)

Script: `rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py`

Flow: **per-episode retrieve + augment → write temp `*_aug_*.json.gz` → patch Habitat / eval config → `internnav.evaluator.Evaluator`**.

```bash
python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py \
  --config rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py \
  --augmenter semantic_pathplanning \
  --text-embedder bge --vision-embedder vit \
  --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt \
  --max-episodes 5 \
  --save-instruction-pairs
```

- **`--max-episodes 0` (or negative):** process and evaluate the full split.
- **KB embed cache (~8 GB):** reuse after first build; saves ~2–3 minutes per episode.
- **Offline / no API:** add `--no-robot-image` to skip VLM caption.
- **`conda run`:** add `--no-capture-output` for live logs.

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | InternNav eval `.py` config | `rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py` |
| `--augmenter` | See §1.3 (includes `r_only`) | `semantic_pathplanning` |
| `--rag4vln-config` | Unified YAML | `rag4vln/src/config.yaml` |
| `--kb-root` | KB root | `rag4vln/data/kb/memory` |
| `--text-embedder` / `--vision-embedder` | Same as retrieval demo | `binary` / `binary` |
| `--topk1` / `--topk2` / `--topk3` | Retrieval depth | `3` / `3` / `3` |
| `--max-episodes` | First N if `>0`; all if `<=0` | `1` |
| `--cache-path` | Augmentation result cache JSON | none |
| `--robot-image` | Fixed observation; else per-episode `start_view` | none |
| `--kb-embed-cache` | KB embedding cache | none |
| `--save-instruction-pairs` | Write original vs augmented JSONL | off |
| `--save-video` | Save InternNav eval videos | off |

JSONL fields with `--save-instruction-pairs`: `original_instruction_text`, `augmented_instruction_text`.

More commands: `rag4vln/instruction.md`.

---

## 6. Minimal adaptation for other models

The contract with downstream VLN is: **a Habitat-readable episode dataset (json.gz) whose `instruction.instruction_text` has been replaced with the augmented sentence**. When adding a new backbone, prefer changing only `scripts/eval/` glue, not `src/` core logic.

### 6.1 Recommended two-step flow

1. **Augment only** (reuse or copy the first half of `eval_rag4vln_vln_augmented.py`)  
   Per episode: `Retriever.retrieve` → `Augmenter.augment` → write `instruction_text` into gzip.  
   Example artifact: `rag4vln/results/augmented_vln_eval/<run>/val_unseen_aug_<ts>.json.gz`.

2. **Run the target framework’s Habitat eval**  
   Point `habitat.dataset.data_path` at that gzip; align `scenes_dir` and split with your `data/` layout.  
   No need to import the target model’s training code inside this repo.

### 6.2 InternNav-style: single “augment + eval” script

See `scripts/eval/eval_rag4vln_vln_augmented.py`:

| Step | Practice |
|------|----------|
| Paths | `repo_root` contains `rag4vln/` and downstream package; add `repo_root` and `rag4vln/` to `sys.path` |
| Augment loop | Reuse `KnowledgeBase`, `Retriever`, `_build_augmenter()`; load per-episode `start_view` images |
| Write dataset | Clone gzip; set `episodes[i].instruction.instruction_text = augmented` |
| Invoke eval | Temp Habitat YAML (patch `data_path` only); call downstream `Evaluator` |
| Isolate outputs | Fresh `output_path` each run to avoid `progress.json` skipping episodes |

**InternNav only needs** an eval cfg (e.g. `scripts/eval/configs/habitat_dual_system_cfg.py`) via `--config`.

### 6.3 StreamVLN-style: subprocess to upstream eval

Full example: `rag4vln/scripts/eval/eval_rag4vln_streamvln.py` in some InternNav forks (copy if missing here). Key points:

- Pass augmented gzip as `--episode-json-gz` / `data_path`.
- From StreamVLN root: `torchrun … streamvln/streamvln_eval.py`; `PYTHONPATH` includes InternNav and StreamVLN roots.
- Optionally mirror StreamVLN `result.json` to `internnav_output/result.json` for comparison.

### 6.4 Add a new instruction augmenter

1. Subclass `InstructionAugmenter` in `src/augment/` and implement `augment(...)`.
2. Export `build_*_augmenter` in `src/augment/__init__.py`.
3. Register in `_build_augmenter()` and argparse `choices` in `eval_rag4vln_vln_augmented.py`.
4. Add a config block under `augment:` in `src/config.yaml`.

### 6.5 Call core APIs from Python

```python
from pathlib import Path
from src.kb import KnowledgeBase
from src.retrieval import Retriever, build_text_embedder_from_config, ViTEmbedder
from src.augment import build_semantic_pathplanning_augmenter, retrieval_evidence_from_plan

cfg = Path("rag4vln/src/config.yaml")
kb = KnowledgeBase(Path("rag4vln/data/kb/memory"))
retriever = Retriever(
    text_embedder=build_text_embedder_from_config(cfg, backend="bge"),
    vision_embedder=ViTEmbedder(config_path=cfg),
    caption_config_path=cfg,
)
augmenter = build_semantic_pathplanning_augmenter(config_path=cfg)

plan = retriever.retrieve(kb, instruction="go to the TV", robot_image_path="path/to/start.png")
evidence = retrieval_evidence_from_plan(plan)
result = augmenter.augment("go to the TV", evidence)
print(result.instruction)  # augmented navigation instruction
```

Write `result.instruction` into your dataloader or json.gz to plug into any VLN model.
