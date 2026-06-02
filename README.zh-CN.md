# rag4vln

**Languages:** [English](README.md) · **简体中文**

> 以本文件为主修改，定稿后同步 [README.md](README.md)。

面向 VLN 的 **RAG 插件**：根据用户意图与起始观测，从记忆库检索证据并增强导航指令，再交给 InternNav、StreamVLN 等下游做 Habitat 评测。核心逻辑与具体 VLN 模型**解耦**。

**运行约定**：命令均在 **InternNav 仓库根目录**执行（含 `rag4vln/` 与 `internnav/`）。更多命令见 [`instruction.md`](instruction.md)。

---

## 一、项目介绍

rag4vln 做两件事：**检索**（scene / zone / view）→ **指令增强**（`instruction_text`），输出增强后的 `json.gz` 供下游评测；不训练导航策略。

```
用户指令 + start_view → 检索 → 增强 → 临时 dataset → 下游 VLN Habitat 评测
```

**目录摘要**

| 路径 | 作用 |
|------|------|
| `src/` | 检索、增强、KB、`config.yaml` |
| `scripts/demo/` | 检索 / 增强试跑 |
| `scripts/eval/` | 检索指标、增强 + InternNav 评测 |
| `data/` | memory、KB 等（大文件不进 git，见 §三） |

---

## 二、安装

推荐在 **同一 Conda 环境**（示例 `inter_hab`）中完成下列步骤。之后所有命令均在 **InternNav 仓库根目录**执行。

### 2.1 安装 InternNav + Habitat

按 [InternNav 官方安装文档](https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html) 配置环境，核心步骤示例：

```bash
git clone https://github.com/InternRobotics/InternNav.git --recursive
cd InternNav

conda create -n inter_hab python=3.10 -y
conda activate inter_hab

conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
cd habitat-lab && pip install -e habitat-lab && pip install -e habitat-baselines && cd ..

pip install -e ".[habitat,internvla_n1]"   # 仅做检索离线评测时可只装 [habitat]
```

在 **InternNav 根目录**验证 Habitat 是否可用：

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_s2_cfg.py
```

### 2.2 克隆 rag4vln 并安装额外依赖

在 InternNav 根目录下将本仓库放到 `rag4vln/` 子目录（与 `internnav/` 同级）：

```bash
cd /path/to/InternNav
git clone https://github.com/juanjuanjuanbing/rag4vln.git rag4vln
```

在同一 Conda 环境中安装 rag4vln 所需包：

```bash
pip install numpy pyyaml tqdm pillow torch transformers sentence-transformers openai
export DASHSCOPE_API_KEY="your-key"   # caption / LLM 增强；api_key 见 src/config.yaml
```

离线试跑（不调 VLM、不调用 LLM）：`--text-embedder binary --vision-embedder binary`、`--no-robot-image`；评测可用 `--augmenter r_only`。

---

## 三、数据准备

`rag4vln/data/` 在 git 中**只有目录占位**（`.gitkeep`），大文件需本地解压或下载填入，详见 `rag4vln/data/README.md`。

### 3.1 下载场景与官方 R2R 指令（InternNav）

按 [InternNav 仿真数据说明](https://internrobotics.github.io/user_guide/internnav/quick_start/simulation.html) 下载 MP3D 场景等资源，解压到 InternNav 根目录 `data/`（路径见下表「官方」两行）。

### 3.2 下载 rag4vln 数据整合包

下载 [rag4vln 数据整合包（Google Drive）](https://drive.google.com/file/d/1TZaNuR4r4LbalCAP5a3DV0xKMHjc6LiC/view?usp=drive_link)（含 **NavRAG memory** 与 **评测指令变体**），解压后对齐到：

| 整合包内路径 | 放到 |
|-------------|------|
| `memory/instruction_generator/` | `rag4vln/data/memory/instruction_generator/` |
| `vln_ce/`（或包内等价目录） | InternNav 根目录 `data/vln_ce/` |

勿用整合包覆盖你已自备的 `data/vln_ce/raw_data/`，除非有意替换。整合包**不含** MP3D 场景 mesh 与完整官方 R2R 原始包。

### 3.3 路径一览

**InternNav 根目录 `data/`**

| 内容 | 路径 | 来源 |
|------|------|------|
| MP3D 场景 | `data/scene_data/mp3d_ce/mp3d/` | 官方 / InternNav 文档 |
| R2R 原始指令 | `data/vln_ce/raw_data/r2r/<split>/*.json.gz` | 官方 / InternNav 文档 |
| 评测指令变体 | `data/vln_ce/raw_data_mask_1/`、`raw_data_implicit/` 等 | 整合包 |
| GT / 起点图（可选） | `data/vln_ce/dataset_gt.csv`、`start_view/...` | 脚本生成或整合包 |

**`rag4vln/data/`**

| 内容 | 路径 |
|------|------|
| NavRAG memory | `rag4vln/data/memory/instruction_generator/` |
| 构建后的 KB | `rag4vln/data/kb/memory/` |

### 3.4 构建 KB 与检索 GT

在 InternNav 根目录执行（需已完成 §3.1 场景与 §3.2 memory）：

```bash
# 构建 KB（可选 --render-view-images 渲染视角图）
python rag4vln/scripts/build_memory_kb.py \
  --memory-dir rag4vln/data/memory/instruction_generator \
  --output-dir rag4vln/data/kb/memory \
  --render-view-images \
  --scene-root data/scene_data/mp3d_ce/mp3d

# 检索评测用 GT 与 start_view 图
python rag4vln/scripts/build_dataset_gt.py
# 隐式指令集：python rag4vln/scripts/build_dataset_gt.py --vln-subdir raw_data_implicit
```

---

## 四、快速开始

```bash
# 仅检索
python rag4vln/scripts/demo/test_retriever_demo.py --text-embedder bge --vision-embedder vit

# 检索 + 增强
python rag4vln/scripts/demo/test_augmenter_demo.py \
  --augmenter semantic_pathplanning --text-embedder bge --vision-embedder vit
```

轻量试跑：`--text-embedder binary --vision-embedder binary`（无需大模型与 API）。完整配置见 `rag4vln/src/config.yaml`。

### 4.1 `test_retriever_demo.py`

| 参数 | 说明 | 默认 |
|------|------|------|
| `--text-embedder` | 文本嵌入：`auto` / `bert` / `sbert` / `bge` / `binary` | `auto` |
| `--vision-embedder` | 图像嵌入：`vit` / `binary` | `vit` |
| `--config` | 统一 YAML（`retrieval` 段） | `rag4vln/src/config.yaml` |
| `--kb-root` | KB 根目录 | `rag4vln/data/kb/memory` |
| `--instruction` | 查询用自然语言指令 | 内置英文示例句 |
| `--robot-image` | 机器人观测图（用于 VLM caption + 检索） | `data/test_materials/test.png` |
| `--no-robot-image` | 不传图，跳过 VLM caption | 关闭 |
| `--binary-dim` | `binary` 嵌入维度 | `64` |
| `--result-dir` | 输出目录 | `rag4vln/results` |
| `--no-save-result` | 不写入 `plan.json` 等 | 关闭 |

### 4.2 `test_augmenter_demo.py`

在 4.1 基础上增加：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--augmenter` | `llm_direct` / `template_path` / `semantic_pathplanning` | `llm_direct` |
| `--config` | 同时读取 `retrieval` 与 `augment` | 同 4.1 |
| `--episode-id` | 从 `--gt-csv` 按 id 读取起点图与指令 | 无 |
| `--gt-csv` | GT 表（含 `start_view_image_path`） | `data/vln_ce/dataset_gt.csv` |
| `--start-view-image` | 起始视角图（优先于 `--robot-image`） | 无 |
| `--no-save-result` | 不保存 `plan.json` / `evidence.json` / `augmentation.json` | 关闭 |

成功运行后可在 `rag4vln/results/<timestamp>/` 查看检索与增强结果。

---

## 五、评测

**检索评测**

```bash
python rag4vln/scripts/eval/eval_retriever.py \
  --dataset-json data/vln_ce/raw_data_mask_1/r2r/val_seen/val_seen.json \
  --gt-csv data/vln_ce/dataset_gt.csv \
  --text-embedder bge --vision-embedder vit \
  --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt \
  --max-episodes 150 --no-export-images
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--dataset-json` | VLN-CE episode JSON（**必填**） | — |
| `--gt-csv` | GT 表，按 `episode_id` 对齐 | `data/vln_ce/dataset_gt.csv` |
| `--subset-name` | 当前子集标签（写入结果目录名） | `full_instruction` |
| `--text-embedder` / `--vision-embedder` | 同 demo | `bge` / `vit` |
| `--topk1` / `--topk2` / `--topk3` | 场景 / 区域 / (start,end) 候选列表长度 | `3` / `3` / `10` |
| `--hit-k` | 指标 `Hit@K` 的 K | `5` |
| `--max-episodes` | `>0` 只评前 N 条；`<=0` 全部 | `0` |
| `--kb-embed-cache` | KB 嵌入缓存 `.pt`，复用可大幅提速 | 无 |
| `--rebuild-kb-embed-cache` | 强制重建上述缓存 | 关闭 |
| `--no-export-images` | 不导出起点/检索对比图 | 关闭 |
| `--result-dir` | 评测输出根目录 | `rag4vln/results/retriever_eval` |

需事先存在 `data/vln_ce/start_view/.../ep_<id>.png`（由 `build_dataset_gt.py` 生成）。关心 `Hit@K` 时请保证 `--topk3 >= K`。

**增强 + InternNav Habitat 评测**

```bash
python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py \
  --config rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py \
  --augmenter semantic_pathplanning \
  --text-embedder bge --vision-embedder vit \
  --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt \
  --max-episodes 5 --save-instruction-pairs
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | InternNav 评测配置 `.py`（相对仓库根或绝对路径） | `rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py` |
| `--augmenter` | `llm_direct` / `template_path` / `semantic_pathplanning` / `r_only`（无 LLM，仅拼接证据） | `semantic_pathplanning` |
| `--rag4vln-config` | rag4vln 统一 YAML | `rag4vln/src/config.yaml` |
| `--kb-root` | KB 根目录 | `rag4vln/data/kb/memory` |
| `--text-embedder` / `--vision-embedder` | 同 demo | `binary` / `binary` |
| `--topk1` / `--topk2` / `--topk3` | 检索深度（评测常用较小 topk3） | `3` / `3` / `3` |
| `--max-episodes` | `>0` 前 N 条；`<=0` **全部** episode | `1` |
| `--robot-image` | 固定起始图；不传则用每条 episode 的 start_view | 无 |
| `--no-robot-image` | 跳过 VLM caption，适合离线 | 关闭 |
| `--kb-embed-cache` | KB 嵌入缓存（约 8GB，首次构建后复用） | 无 |
| `--rebuild-kb-embed-cache` | 强制重建 KB 缓存 | 关闭 |
| `--save-instruction-pairs` | 写出原句 / 增强句 JSONL | 关闭 |
| `--instruction-pairs-path` | JSONL 路径（默认在当前 run 目录） | 自动 |
| `--save-video` | 保存 InternNav 评测视频 | 关闭 |
| `--no-export-images` | 不导出 ins/retriever/gt 对比图 | 关闭 |
| `--gt-csv` | 开启导出时用于 GT 起终点图 | `data/vln_ce/dataset_gt.csv` |

`--max-episodes 0` 表示增强并评测当前 split 的**全部** episode。长任务建议：`conda run --no-capture-output -n inter_hab python rag4vln/scripts/eval/...`。

---

## 六、适配其他模型

下游只需一份 **`instruction.instruction_text` 已增强** 的 Habitat `json.gz`：

1. 用 `eval_rag4vln_vln_augmented.py` 前半段逻辑，或 demo 中的检索 + 增强，写出增强 gzip；
2. 将目标框架的 `habitat.dataset.data_path` 指向该文件即可。

仿 InternNav 单脚本集成见 `scripts/eval/eval_rag4vln_vln_augmented.py`；StreamVLN 可参考 InternNav 仓库中的 `eval_rag4vln_streamvln.py`。
