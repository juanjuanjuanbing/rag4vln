# rag4vln

**Languages:** **English** · [简体中文](README.zh-CN.md)

> Edit [README.zh-CN.md](README.zh-CN.md) first, then sync this file.

A **RAG plugin** for VLN: retrieve evidence from a memory KB using user intent and the starting observation, augment navigation instructions, then run Habitat evaluation with InternNav, StreamVLN, or other downstream frameworks. Core logic is **decoupled** from any specific VLN model.

**Working directory:** Run commands from the **InternNav repository root** (contains `rag4vln/` and `internnav/`). More commands: [`instruction.md`](instruction.md).

---

## 1. Introduction

rag4vln performs **retrieval** (scene / zone / view) then **instruction augmentation** (`instruction_text`), producing augmented `json.gz` for downstream eval—it does not train navigation policies.

```
User instruction + start_view → retrieve → augment → temp dataset → downstream VLN Habitat eval
```

**Layout**

| Path | Role |
|------|------|
| `src/` | Retrieval, augmentation, KB, `config.yaml` |
| `scripts/demo/` | Retrieval / augmentation trials |
| `scripts/eval/` | Retrieval metrics, augment + InternNav eval |
| `data/` | memory, KB, etc. (large files not in git; see §3) |

---

## 2. Installation

Use one Conda env (e.g. `inter_hab`) for all steps below. Run commands from the **InternNav repository root** afterward.

### 2.1 Install InternNav + Habitat

Follow the [InternNav installation guide](https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html). Example:

```bash
git clone https://github.com/InternRobotics/InternNav.git --recursive
cd InternNav

conda create -n inter_hab python=3.10 -y
conda activate inter_hab

conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
cd habitat-lab && pip install -e habitat-lab && pip install -e habitat-baselines && cd ..

pip install -e ".[habitat,internvla_n1]"   # retrieval-only eval: [habitat] alone is enough
```

Verify Habitat at the **InternNav repo root**:

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_s2_cfg.py
```

### 2.2 Clone rag4vln and install extras

Place this repo under `rag4vln/` next to `internnav/`:

```bash
cd /path/to/InternNav
git clone https://github.com/juanjuanjuanbing/rag4vln.git rag4vln
```

In the same Conda env:

```bash
pip install numpy pyyaml tqdm pillow torch transformers sentence-transformers openai
export DASHSCOPE_API_KEY="your-key"   # caption / LLM augment; see src/config.yaml
```

Offline trials: `--text-embedder binary --vision-embedder binary`, `--no-robot-image`; eval can use `--augmenter r_only`.

---

## 3. Data preparation

`rag4vln/data/` in git contains **placeholders only** (`.gitkeep`); fill large files locally—see `rag4vln/data/README.md`.

### 3.1 Scenes and official R2R (InternNav)

Download MP3D scenes per the [InternNav simulation data guide](https://internrobotics.github.io/user_guide/internnav/quick_start/simulation.html) into InternNav root `data/` (see “official” rows in §3.3).

### 3.2 rag4vln data bundle

Download the [rag4vln data bundle (Google Drive)](https://drive.google.com/file/d/1TZaNuR4r4LbalCAP5a3DV0xKMHjc6LiC/view?usp=drive_link) (**NavRAG memory** + **eval instruction variants**), then map:

| In bundle | Place at |
|-----------|----------|
| `memory/instruction_generator/` | `rag4vln/data/memory/instruction_generator/` |
| `vln_ce/` (or equivalent) | InternNav root `data/vln_ce/` |

Do not overwrite your own `data/vln_ce/raw_data/` unless intended. The bundle does **not** include MP3D meshes or the full official R2R package.

### 3.3 Path reference

**InternNav root `data/`**

| Content | Path | Source |
|---------|------|--------|
| MP3D scenes | `data/scene_data/mp3d_ce/mp3d/` | Official / InternNav docs |
| R2R full instructions | `data/vln_ce/raw_data/r2r/<split>/*.json.gz` | Official / InternNav docs |
| Eval instruction variants | `data/vln_ce/raw_data_mask_1/`, `raw_data_implicit/`, etc. | Data bundle |
| GT / start views (optional) | `data/vln_ce/dataset_gt.csv`, `start_view/...` | Script or bundle |

**`rag4vln/data/`**

| Content | Path |
|---------|------|
| NavRAG memory | `rag4vln/data/memory/instruction_generator/` |
| Built KB | `rag4vln/data/kb/memory/` |

### 3.4 Build KB and retrieval GT

At InternNav repo root (after §3.1 scenes and §3.2 memory):

```bash
# Build KB (optional --render-view-images for view PNGs)
python rag4vln/scripts/build_memory_kb.py \
  --memory-dir rag4vln/data/memory/instruction_generator \
  --output-dir rag4vln/data/kb/memory \
  --render-view-images \
  --scene-root data/scene_data/mp3d_ce/mp3d

# GT table & start_view images for retrieval eval
python rag4vln/scripts/build_dataset_gt.py
# Implicit set: python rag4vln/scripts/build_dataset_gt.py --vln-subdir raw_data_implicit
```

---

## 4. Quick start

```bash
# Retrieval only
python rag4vln/scripts/demo/test_retriever_demo.py --text-embedder bge --vision-embedder vit

# Retrieval + augmentation
python rag4vln/scripts/demo/test_augmenter_demo.py \
  --augmenter semantic_pathplanning --text-embedder bge --vision-embedder vit
```

Lightweight run: `--text-embedder binary --vision-embedder binary` (no large models or API). Full config: `rag4vln/src/config.yaml`.

### 4.1 `test_retriever_demo.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--text-embedder` | Text embedding: `auto` / `bert` / `sbert` / `bge` / `binary` | `auto` |
| `--vision-embedder` | Image embedding: `vit` / `binary` | `vit` |
| `--config` | Unified YAML (`retrieval` section) | `rag4vln/src/config.yaml` |
| `--kb-root` | KB root | `rag4vln/data/kb/memory` |
| `--instruction` | Query instruction text | Built-in English example |
| `--robot-image` | Robot observation (VLM caption + retrieval) | `data/test_materials/test.png` |
| `--no-robot-image` | Skip image and VLM caption | off |
| `--binary-dim` | `binary` embedding dimension | `64` |
| `--result-dir` | Output directory | `rag4vln/results` |
| `--no-save-result` | Do not write `plan.json`, etc. | off |

### 4.2 `test_augmenter_demo.py`

Adds on top of §4.1:

| Argument | Description | Default |
|----------|-------------|---------|
| `--augmenter` | `llm_direct` / `template_path` / `semantic_pathplanning` | `llm_direct` |
| `--config` | Reads both `retrieval` and `augment` | same as §4.1 |
| `--episode-id` | Load start image & instruction from `--gt-csv` by id | none |
| `--gt-csv` | GT table (with `start_view_image_path`) | `data/vln_ce/dataset_gt.csv` |
| `--start-view-image` | Start-view image (overrides `--robot-image`) | none |
| `--no-save-result` | Skip `plan.json` / `evidence.json` / `augmentation.json` | off |

Outputs land under `rag4vln/results/<timestamp>/`.

---

## 5. Evaluation

**Retrieval eval**

```bash
python rag4vln/scripts/eval/eval_retriever.py \
  --dataset-json data/vln_ce/raw_data_mask_1/r2r/val_seen/val_seen.json \
  --gt-csv data/vln_ce/dataset_gt.csv \
  --text-embedder bge --vision-embedder vit \
  --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt \
  --max-episodes 150 --no-export-images
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset-json` | VLN-CE episode JSON (**required**) | — |
| `--gt-csv` | GT table, keyed by `episode_id` | `data/vln_ce/dataset_gt.csv` |
| `--subset-name` | Subset tag (used in output dir name) | `full_instruction` |
| `--text-embedder` / `--vision-embedder` | Same as demo | `bge` / `vit` |
| `--topk1` / `--topk2` / `--topk3` | Scene / zone / (start,end) list depth | `3` / `3` / `10` |
| `--hit-k` | K for `Hit@K` metrics | `5` |
| `--max-episodes` | First N if `>0`; all if `<=0` | `0` |
| `--kb-embed-cache` | KB embedding cache `.pt` (large speedup on reuse) | none |
| `--rebuild-kb-embed-cache` | Force rebuild cache | off |
| `--no-export-images` | Skip start/retrieval comparison images | off |
| `--result-dir` | Output root | `rag4vln/results/retriever_eval` |

Requires `data/vln_ce/start_view/.../ep_<id>.png` from `build_dataset_gt.py`. Use `--topk3 >= K` when reporting `Hit@K`.

**Augment + InternNav Habitat eval**

```bash
python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py \
  --config rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py \
  --augmenter semantic_pathplanning \
  --text-embedder bge --vision-embedder vit \
  --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt \
  --max-episodes 5 --save-instruction-pairs
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | InternNav eval `.py` (relative to repo root or absolute) | `rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py` |
| `--augmenter` | `llm_direct` / `template_path` / `semantic_pathplanning` / `r_only` (no LLM) | `semantic_pathplanning` |
| `--rag4vln-config` | rag4vln unified YAML | `rag4vln/src/config.yaml` |
| `--kb-root` | KB root | `rag4vln/data/kb/memory` |
| `--text-embedder` / `--vision-embedder` | Same as demo | `binary` / `binary` |
| `--topk1` / `--topk2` / `--topk3` | Retrieval depth | `3` / `3` / `3` |
| `--max-episodes` | First N if `>0`; **all** episodes if `<=0` | `1` |
| `--robot-image` | Fixed start image; else per-episode `start_view` | none |
| `--no-robot-image` | Skip VLM caption (offline-friendly) | off |
| `--kb-embed-cache` | KB embed cache (~8GB; reuse after first build) | none |
| `--rebuild-kb-embed-cache` | Force rebuild cache | off |
| `--save-instruction-pairs` | Write original vs augmented JSONL | off |
| `--instruction-pairs-path` | JSONL path (default: current run dir) | auto |
| `--save-video` | Save InternNav eval videos | off |
| `--no-export-images` | Skip ins/retriever/gt image export | off |
| `--gt-csv` | GT table when image export is enabled | `data/vln_ce/dataset_gt.csv` |

`--max-episodes 0` augments and evaluates the **full** split. Long jobs: `conda run --no-capture-output -n inter_hab python rag4vln/scripts/eval/...`.

---

## 6. Adapting other models

Downstream only needs a Habitat `json.gz` with **augmented** `instruction.instruction_text`:

1. Run retrieval + augmentation (first half of `eval_rag4vln_vln_augmented.py` or the demos) and write the augmented gzip;
2. Point the target framework’s `habitat.dataset.data_path` at that file.

Single-script InternNav integration: `scripts/eval/eval_rag4vln_vln_augmented.py`. StreamVLN: see `eval_rag4vln_streamvln.py` in some InternNav forks.
