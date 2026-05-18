# rag4vln

**Languages:** [English](README.md) · **简体中文**

面向视觉语言导航（VLN）的 **RAG 插件**：在下游导航模型评测前，根据用户意图与机器人起始观测，从记忆库（KB）检索场景/区域/视角证据，再经可插拔的指令增强器生成更可执行的导航指令。核心检索与增强逻辑与具体 VLN backbone **解耦**；通过 `scripts/eval/` 下的胶水脚本对接 InternNav、StreamVLN 等框架。

**运行约定**：除单独说明外，命令均在 **InternNav 仓库根目录**执行（目录下同时包含 `rag4vln/` 与 `internnav/`）。

---

## 一、项目介绍

### 1.1 简介

rag4vln 在标准 VLN 流水线中插入两个阶段：

1. **检索**：`Retriever` 在 KB 中匹配场景（scene）、区域（zone）、起终点视角（view），并给出 zone-graph 上的路径证据。
2. **指令增强**：`InstructionAugmenter` 将不完整/隐式指令与检索证据合成为完整、可执行的 `instruction_text`。

下游 Habitat 评测只需读取**增强后**的数据集；本仓库不负责训练导航策略，只负责「检索 + 改写指令」。

**当前约定**

- KB 为 `scene` → `zone` → `view` 三层，位于 `rag4vln/data/kb/<name>/scenes/*.json`。
- 多个 viewpoint 在构建时已**扁平化**为独立 `view`（ID 不重复）。
- 检索路径基于 **zone graph**，不依赖 view graph。
- `instances`（view 内物品细节）不作为核心结构。

**流水线（与下游无关部分）**

```
用户指令 + start_view 图 → Retriever → 结构化证据 → Augmenter → 增强后指令
                                                              ↓
                                                    临时 dataset (.json.gz)
                                                              ↓
                                                    下游 VLN 模型 Habitat 评测
```

### 1.2 文件组织

```
rag4vln/
├── src/                          # 核心 Python 包（可被 demo / eval 直接 import）
│   ├── config.yaml               # 统一配置：basic / retrieval / augment
│   ├── config_io.py              # 配置加载（兼容旧版扁平 YAML）
│   ├── kb/                       # 知识库构建与读取
│   ├── retrieval/                # 嵌入、caption、检索打分
│   ├── augment/                  # 指令增强器（可插拔）
│   └── utils/                    # Habitat 渲染、IO 等工具
├── data/
│   ├── kb/                       # 构建好的 KB（scenes、imgs、manifest）
│   ├── memory/                   # KB 构建源数据（标注、connectivity）
│   └── test_materials/           # demo 用示例图
├── scripts/
│   ├── demo/                     # 检索 / 增强快速试跑
│   ├── eval/                     # 检索指标评测、增强 + 下游 VLN 评测
│   │   └── configs/              # InternNav eval / Habitat dataset 配置
│   ├── build_memory_kb.py        # KB 构建入口
│   └── build_dataset_gt.py       # 检索评测用 GT 与 start_view 图
├── results/                      # 运行输出（建议 gitignore）
├── README.md
├── README.zh-CN.md
└── instruction.md                # 常用命令速查
```

### 1.3 代码布局

| 路径 | 作用 |
|------|------|
| `src/kb/kb.py` | `KnowledgeBase`：读取 scene JSON |
| `src/kb/kb_build.py` | 从 memory 数据构建 KB，可选渲染视角图 |
| `src/retrieval/retriever.py` | `Retriever.retrieve`：分层检索与 zone 路径 |
| `src/retrieval/embedder.py` | 文本/图像嵌入（BERT、SBERT、BGE、ViT、binary） |
| `src/retrieval/caption.py` | 机器人观测 VLM caption（DashScope 等） |
| `src/augment/instruction_augmenter.py` | 增强器抽象基类 |
| `src/augment/*_augmenter.py` | 具体增强策略实现 |
| `scripts/demo/test_retriever_demo.py` | 仅检索 demo |
| `scripts/demo/test_augmenter_demo.py` | 检索 + 指令增强 demo |
| `scripts/eval/eval_retriever.py` | 检索离线指标（Hit@K、MRR） |
| `scripts/eval/eval_rag4vln_vln_augmented.py` | 增强后接 **InternNav** Habitat 评测 |

**Scene JSON（简化）**

- `scene.attributes`：`description`、`zone_ids`、`zone_graph.adjacency`、`view_ids`
- `zones[zone_id].attributes`：`description`、`scene_id`、`adjacent_zone_ids`、`view_ids`
- `views[view_id].attributes`：`description`、`position`、`rotation`、`zone_id`、`img`、`included`

**`Retriever.retrieve` 返回要点**

- `topk1_scenes`、`topk2_zones`、`topk3_pairs`
- `topk3_pairs` 常见字段：`start_zone_id`、`start_view_id`、`end_zone_id`、`end_view_id`、`scores`、`path_zone_ids`

**指令增强器（`--augmenter`）**

| 名称 | 说明 |
|------|------|
| `llm_direct` | 单次 LLM 直接扩写 |
| `template_path` | LLM 填槽 + 本地模板拼接 |
| `semantic_pathplanning` | 三阶段 CoT + 最终 VLN 句式（评测默认） |
| `r_only` | 仅拼接检索证据与原文，无 LLM（baseline） |

---

## 二、安装

推荐在 **同一 Conda 环境** 中完成 InternNav Habitat 与 rag4vln 依赖安装，以便直接跑 §五 的 Habitat 端到端评测。下文以环境名 `inter_hab` 为例（与评测脚本提示一致，可自定）。

### 2.1 安装 InternNav 的 Habitat 环境

rag4vln 的 VLN-CE 评测依赖 **InternNav + 内置 Habitat-Lab**。请先在 **InternNav 仓库根目录**（含 `internnav/`、`habitat-lab/`、`rag4vln/`）按官方流程配置仿真环境。

**1. 获取代码**

```bash
git clone https://github.com/InternRobotics/InternNav.git --recursive
cd InternNav
# 若 rag4vln 为独立目录，请置于本仓库根下：InternNav/rag4vln/
```

**2. 创建 Conda 环境**

```bash
conda create -n inter_hab python=3.10 -y
conda activate inter_hab
```

Python 版本以 [InternNav 安装文档](https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html) 为准（通常 3.9–3.10）。

**3. 安装 Habitat-Sim**

```bash
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

版本需与仓库内 `habitat-lab` 兼容；若官方文档指定了 `habitat-sim==x.y`，请以其为准。

**4. 安装仓库自带的 Habitat-Lab / Baselines**

```bash
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
```

**5. 安装 InternNav（Habitat + 评测模型依赖）**

```bash
pip install -e ".[habitat,internvla_n1]"
```

说明：

- `[habitat]`：VLN-CE 仿真与 `internnav.habitat_extensions` 所需依赖（见 `requirements/habitat_requirements.txt`）。
- `[internvla_n1]`：跑 `eval_rag4vln_vln_augmented.py` 默认双系统配置时的模型相关依赖（见 `requirements/internvla_n1.txt`）。若仅做检索离线评测、不启动 InternVLA，可只装 `[habitat]`。

更完整的 GPU/CUDA、checkpoint 与可选组件说明，见 InternNav 用户指南：[Quick Start — Installation](https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html)。

**6. 自检（可选）**

```bash
python -c "import habitat; import internnav; print('habitat + internnav OK')"
```

### 2.2 安装 rag4vln 所需环境

rag4vln **无独立 `setup.py`**，在已激活的 `inter_hab`（或你的 InternNav 环境）中追加 Python 包即可；核心代码通过 `sys.path` 挂载 `rag4vln/` 目录。

**1. 检索与评测常用依赖**

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

| 用途 | 主要依赖 |
|------|----------|
| 检索嵌入 `bge` / `vit` | `torch`, `transformers`, `sentence-transformers` |
| 检索嵌入 `binary`（demo 快速试跑） | 仅需 `numpy` |
| VLM caption + LLM 指令增强 | `openai`（兼容 DashScope 等 OpenAI 格式 API）、`pillow` |
| KB 构建 / `build_dataset_gt` 渲染 | 同上 + §2.1 的 **Habitat-Sim** |
| 脚本与配置 | `pyyaml`, `tqdm` |

首次使用 `bge` / `vit` 时会从 Hugging Face 拉取权重，请保证网络或已配置镜像。

**2. 配置 API（caption 与增强）**

编辑 `rag4vln/src/config.yaml` 中 `retrieval.caption` 与 `augment.*` 的 `api_key_env` / `base_url` / `model`；推荐将密钥写入环境变量，例如：

```bash
export DASHSCOPE_API_KEY="your-key"   # 示例：阿里云 DashScope
```

离线试跑检索（不调 VLM、不调用 LLM）时，可使用 demo 的 `--no-robot-image` 与 `--augmenter r_only`（仅评测脚本），或 `--text-embedder binary --vision-embedder binary`。

**3. 自检（可选）**

在 InternNav 仓库根目录：

```bash
python rag4vln/scripts/demo/test_retriever_demo.py \
  --text-embedder binary --vision-embedder binary --no-save-result
```

若 §三 中 KB 已就绪且需验证语义嵌入：

```bash
python rag4vln/scripts/demo/test_retriever_demo.py \
  --text-embedder bge --vision-embedder vit --no-save-result
```

**4. 使用 `conda run` 跑长任务**

长时间评测建议实时输出日志：

```bash
conda run --no-capture-output -n inter_hab python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py ...
```

---

## 三、数据准备

为跑通官方评测流程，需准备三类数据（路径均相对于 **InternNav 仓库根目录**，即与 `rag4vln/`、`internnav/` 同级）。

```
<repo_root>/
├── data/
│   ├── scene_data/mp3d_ce/          # ① 场景 mesh（InternNav）
│   └── vln_ce/                      # ① 原始 R2R 指令 + ③ 评测指令子集
│       ├── raw_data/
│       ├── raw_data_mask_1/
│       ├── raw_data_implicit/       # 或 raw_data_mask_0.5，视整合包为准
│       ├── start_view/              # 检索评测用起点图（脚本生成）
│       └── dataset_gt.csv
└── rag4vln/
    └── data/
        ├── memory/                  # ② NavRAG memory（整合包提供）
        └── kb/memory/               # 由 memory 构建（见下文）
```

### 3.1 场景与原始指令集（InternNav）

从 **[InternNav](https://github.com/InternRobotics/InternNav)** 按官方文档准备 VLN-CE 所需资源，并放到仓库根下 `data/`：

| 内容 | 放置路径 | 说明 |
|------|----------|------|
| MP3D 场景（VLN-CE 重排） | `data/scene_data/mp3d_ce/mp3d/<scene_id>/` | Habitat 仿真与 KB 视角渲染（`--scene-root`）依赖此目录 |
| R2R 原始 episode（完整指令） | `data/vln_ce/raw_data/r2r/<split>/<split>.json.gz` | 官方 VLN-CE R2R；`split` 如 `val_seen`、`val_unseen` |

下载方式请遵循 InternNav 用户文档中的 **VLN-CE / 场景数据** 章节，例如：

- 场景：InternData-N1 / Scene-N1 中的 MP3D-CE 资源（常见包名 `mp3d_pe` 等，解压到 `data/scene_data/`）
- 指令：Habitat 官方 R2R VLN-CE 压缩包，或 InternNav 文档给出的已转换 `json.gz` 布局

具体 wget / Hugging Face 链接以 [InternNav 安装与数据说明](https://internrobotics.github.io/user_guide/internnav/quick_start/index.html) 为准。

### 3.2 Memory（NavRAG）

检索记忆库由 **NavRAG** 产出的 scene / zone / view 标注与 connectivity 图构建，需解压到：

```text
rag4vln/data/memory/instruction_generator/
```

解压后应至少包含（与 `scripts/build_memory_kb.py` 默认参数一致）：

| 文件 / 目录 | 用途 |
|-------------|------|
| `mp3d_view_annotation.json` | view 级描述与位姿 |
| `mp3d_zone_annotation.json` | zone 划分 |
| `mp3d_house_annotation.json` | 场景级 house 标注（可选） |
| `connectivity_mp3d/` | MP3D 各 scan 的 connectivity JSON |

**单独获取**：按 [NavRAG](https://github.com/MrZihan/NavRAG) 官方仓库说明下载 memory 产物，将 `instruction_generator` 整目录放入 `rag4vln/data/memory/`。

**推荐**：使用下文 **§3.4 整合包**（已含 memory，无需再单独下 NavRAG）。

构建 KB（需已安装 Habitat 且 §3.1 场景就绪；渲染视角图时加 `--render-view-images`）：

```bash
python rag4vln/scripts/build_memory_kb.py \
  --memory-dir rag4vln/data/memory/instruction_generator \
  --output-dir rag4vln/data/kb/memory \
  --render-view-images \
  --scene-root data/scene_data/mp3d_ce/mp3d
```

默认产物：`rag4vln/data/kb/memory/scenes/*.json`，及可选 `imgs/<scene_id>/<view_id>.png`。

### 3.3 评测指令集

用于 **检索评测**（`eval_retriever.py`）与 **增强 + Habitat 评测**（`eval_rag4vln_vln_augmented.py`）的指令变体，放在仓库根：

```text
data/vln_ce/
├── raw_data/              # 完整指令（与 §3.1 原始集一致时可共用）
├── raw_data_mask_1/       # 缺省 / 掩码指令（示例评测常用）
├── raw_data_implicit/     # 隐式目标指令（整合包内命名以实际为准）
└── dataset_gt.csv         # 可选：整合包提供，或由脚本生成
```

各子目录结构与 `raw_data` 相同，例如：

`data/vln_ce/raw_data_mask_1/r2r/val_seen/val_seen.json.gz`

**检索评测额外步骤**：在 KB 与场景就绪后，生成 GT 表与共用起点图：

```bash
python rag4vln/scripts/build_dataset_gt.py
# 隐式集：python rag4vln/scripts/build_dataset_gt.py --vln-subdir raw_data_implicit
```

输出默认：`data/vln_ce/dataset_gt.csv`、`data/vln_ce/start_view/r2r/<split>/ep_<episode_id>.png`。

**单独获取**：若仅缺评测指令变体，可按项目 Releases / 文档下载对应 `vln_ce` 子树；**推荐直接使用 §3.4 整合包**。

### 3.4 数据整合包（memory + 评测指令集）

为减少分散下载，我们提供 **rag4vln 数据整合包**（含 **§3.2 memory** 与 **§3.3 评测指令集**）。

| 步骤 | 操作 |
|------|------|
| 1. 下载 | **（下载链接待补充，见项目 Releases 或 README 更新）** |
| 2. 解压 memory | 将包内 `memory/instruction_generator/` 对齐到 `rag4vln/data/memory/instruction_generator/` |
| 3. 解压评测数据 | 将包内 `vln_ce/` 对齐到仓库根 `data/vln_ce/`（勿覆盖你已自备的 `raw_data`，除非有意替换） |
| 4. 构建 KB | 执行 §3.2 中 `build_memory_kb.py` |
| 5. 生成 GT（若包内无 `dataset_gt.csv`） | 执行 §3.3 中 `build_dataset_gt.py` |

整合包 **不包含** MP3D 场景 mesh 与官方完整 R2R 原始包，仍需按 **§3.1** 从 InternNav / Habitat 渠道自行准备。

---

## 四、快速开始

本节用于在**不跑完整 Habitat 评测**的情况下，快速验证检索与指令增强。通过 `--text-embedder` / `--vision-embedder` / `--augmenter` 指定嵌入与增强策略；需要真实语义时请使用 `bge` + `vit`（依赖 `src/config.yaml` 中的模型与 API 配置）。

### 4.1 仅检索

```bash
python rag4vln/scripts/demo/test_retriever_demo.py --text-embedder bge --vision-embedder vit
```

轻量试跑（随机二值嵌入，无需加载大模型）：

```bash
python rag4vln/scripts/demo/test_retriever_demo.py --text-embedder binary --vision-embedder binary
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--text-embedder` | `auto` \| `bert` \| `sbert` \| `bge` \| `binary` | `auto` |
| `--vision-embedder` | `vit` \| `binary` | `vit` |
| `--config` | 统一 YAML（`retrieval` 段） | `rag4vln/src/config.yaml` |
| `--kb-root` | KB 根目录 | `rag4vln/data/kb/memory` |
| `--instruction` | 查询用自然语言指令 | 内置英文示例句 |
| `--binary-dim` | `binary` 嵌入维度 | `64` |
| `--robot-image` | 机器人观测图（caption + 检索） | `rag4vln/data/test_materials/test.png` |
| `--no-robot-image` | 不传图，跳过 VLM caption | 关闭 |
| `--result-dir` | 结果目录 | `rag4vln/results` |
| `--no-save-result` | 不写入 `plan.json` 等 | 关闭 |

### 4.2 检索 + 指令增强

```bash
python rag4vln/scripts/demo/test_augmenter_demo.py \
  --augmenter semantic_pathplanning \
  --text-embedder bge \
  --vision-embedder vit
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--augmenter` | `llm_direct` \| `template_path` \| `semantic_pathplanning` | `llm_direct` |
| `--text-embedder` | 同 4.1 | `auto` |
| `--vision-embedder` | 同 4.1 | `vit` |
| `--config` | 统一 YAML（`retrieval` + `augment`） | `rag4vln/src/config.yaml` |
| `--robot-image` | 机器人图像 | `rag4vln/data/test_materials/test.png` |
| `--instruction` | 用户意图文本 | 内置英文示例句 |
| `--episode-id` | 从 GT CSV 读取该 episode 的起点图与指令 | 无 |
| `--gt-csv` | 含 `start_view_image_path` 的 GT 表 | `data/vln_ce/dataset_gt.csv` |
| `--binary-dim` | `binary` 嵌入维度 | `64` |
| `--result-dir` | 输出目录 | `rag4vln/results` |
| `--no-save-result` | 不保存 `plan.json` / `evidence.json` / `augmentation.json` | 关闭 |

成功运行后，在 `rag4vln/results/<timestamp>/` 下可查看 `plan.json`（检索结果）、`evidence.json`、`augmentation.json`（增强后指令）。

### 4.3 其他调试脚本（可选）

| 脚本 | 用途 |
|------|------|
| `scripts/demo/visualize_kb_views.py` | 导出 KB 中指定 scene 的视角 PNG 与侧车 JSON |
| `scripts/demo/visualize_connectivity_mp3d.py` | 查看 MP3D connectivity 摘要 |

---

## 五、评测

### 5.1 检索评测

脚本：`rag4vln/scripts/eval/eval_retriever.py`

在带 GT 的 VLN-CE 子集上评估检索质量。核心指标：

- **Scene**：`Hit@1` / `Hit@K`（默认 `K=5`）
- **View**：`Hit@1` / `Hit@K` / `MRR`（分别统计 start / end；GT 不在 `topk3_pairs` 内则 MRR 为 0）

若关心 `Hit@K`，需满足 **`--topk3 >= K`**（默认 `--topk3 10` 与 `--hit-k 5` 对齐）。

**数据集切换**（`--dataset-json`）示例：

- 完全指令：`data/vln_ce/raw_data/r2r/...`
- 缺省指令：`data/vln_ce/raw_data_mask_1/r2r/...`
- 隐式指令：`data/vln_ce/raw_data_implicit/r2r/...`（或 `raw_data_mask_0.5`，视仓库布局而定）

GT 表默认：`data/vln_ce/dataset_gt.csv`（按 **`episode_id`** 对齐；同一 id 多行取首次出现）。起点图路径：`data/vln_ce/start_view/r2r/<split>/ep_<episode_id>.png`（需先运行 `scripts/build_dataset_gt.py` 生成）。

**命令示例**

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

| 参数 | 说明 | 默认 |
|------|------|------|
| `--dataset-json` | VLN_CE 样本 JSON（必填） | — |
| `--gt-csv` | GT 对齐表 | `data/vln_ce/dataset_gt.csv` |
| `--subset-name` | 子集标签（写入输出） | `full_instruction` |
| `--rag4vln-config` | 检索 YAML | `rag4vln/src/config.yaml` |
| `--kb-root` | KB 根目录 | `rag4vln/data/kb/memory` |
| `--text-embedder` | `auto` \| `bert` \| `sbert` \| `bge` \| `binary` | `bge` |
| `--vision-embedder` | `vit` \| `binary` | `vit` |
| `--topk1` / `--topk2` / `--topk3` | 场景 / 区域 / pair 列表深度 | `3` / `3` / `10` |
| `--hit-k` | `Hit@K` 的 K | `5` |
| `--max-episodes` | `>0` 前 N 条；`<=0` 全部 | `0` |
| `--no-export-images` | 不导出对比图 | 关闭 |
| `--kb-embed-cache` | KB 嵌入缓存 `.pt` | 无 |
| `--result-dir` | 输出根目录 | `rag4vln/results/retriever_eval` |

主要输出：`metrics.json`、`details.jsonl`、`result.txt`；未加 `--no-export-images` 时另有 `ins_start_view/`、`retriever_start_view/`、`retriever_end_view/`。

### 5.2 指令增强 + 下游 VLN 评测（InternNav）

脚本：`rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py`

流程：**逐 episode 检索 + 增强 → 写临时 `*_aug_*.json.gz` → patch Habitat / eval 配置 → 调用 `internnav.evaluator.Evaluator`**。

```bash
python rag4vln/scripts/eval/eval_rag4vln_vln_augmented.py \
  --config rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py \
  --augmenter semantic_pathplanning \
  --text-embedder bge --vision-embedder vit \
  --kb-embed-cache rag4vln/results/cache/kb_embed_bge_vit.pt \
  --max-episodes 5 \
  --save-instruction-pairs
```

- **`--max-episodes 0`（或负数）**：处理并评测当前 split 的全部 episode。
- **KB 嵌入缓存**（约 8GB）：首次构建后复用，每条 episode 可节省约 2–3 分钟。
- 离线或无 API：可加 `--no-robot-image`，跳过 VLM caption。
- 使用 `conda run` 时建议加 `--no-capture-output`，避免长时间无日志输出。

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | InternNav eval 配置 `.py` | `rag4vln/scripts/eval/configs/habitat_dual_system_cfg.py` |
| `--augmenter` | 见 §1.3 表（含 `r_only`） | `semantic_pathplanning` |
| `--rag4vln-config` | 统一 YAML | `rag4vln/src/config.yaml` |
| `--kb-root` | KB 根目录 | `rag4vln/data/kb/memory` |
| `--text-embedder` / `--vision-embedder` | 同检索 demo | `binary` / `binary` |
| `--topk1` / `--topk2` / `--topk3` | 检索深度 | `3` / `3` / `3` |
| `--max-episodes` | `>0` 前 N 条；`<=0` 全部 | `1` |
| `--cache-path` | 增强结果缓存 JSON | 无 |
| `--robot-image` | 固定观测图；不传则用各 episode 的 start_view | 无 |
| `--kb-embed-cache` | KB 嵌入缓存 | 无 |
| `--save-instruction-pairs` | 写出原句 / 增强句 JSONL | 关闭 |
| `--save-video` | 保存 InternNav 评测视频 | 关闭 |

`--save-instruction-pairs` 生成的 JSONL 字段：`original_instruction_text`、`augmented_instruction_text`。

更完整的命令示例见 `rag4vln/instruction.md`。

---

## 六、适配其他模型的最小改动

rag4vln 与下游 VLN 的边界是：**输出一份 Habitat 可读的 episode 数据集（json.gz），其中 `instruction.instruction_text` 已替换为增强句**。适配新模型时，尽量只改 `scripts/eval/` 胶水层，不动 `src/` 核心。

### 6.1 推荐两步（最省事）

1. **用现有脚本只做增强**（或仿照 `eval_rag4vln_vln_augmented.py` 前半段）  
   对每条 episode：`Retriever.retrieve` → `Augmenter.augment` → 写回 gzip 中的 `instruction_text`。  
   产物示例：`rag4vln/results/augmented_vln_eval/<run>/val_unseen_aug_<ts>.json.gz`。

2. **用目标框架自带的 Habitat 评测**  
   将 `habitat.dataset.data_path` 指向上一步的 gzip，`scenes_dir` 与 split 与仓库 `data/` 布局一致即可。  
   无需在本仓库内 import 目标模型的训练代码。

### 6.2 仿 InternNav：单脚本「增强 + 评测」

参考 `scripts/eval/eval_rag4vln_vln_augmented.py`：

| 步骤 | 做法 |
|------|------|
| 路径 | `repo_root` = 含 `rag4vln/` 与下游包的根；`sys.path` 插入 `repo_root` 与 `rag4vln/` |
| 增强循环 | 复用 `KnowledgeBase`、`Retriever`、`_build_augmenter()`；按 episode 读 start_view 图 |
| 写数据集 | 复制原 gzip，`episodes[i].instruction.instruction_text = augmented` |
| 调评测 | 生成临时 Habitat YAML（只改 `data_path`），再调用下游 `Evaluator` 或等价入口 |
| 输出隔离 | 每次 run 使用新的 `output_path`，避免 `progress.json` 导致跳过 episode |

**InternNav 仅需**：准备 eval cfg（如 `scripts/eval/configs/habitat_dual_system_cfg.py`），运行时 `--config` 指向该文件。

### 6.3 仿 StreamVLN：子进程调 upstream eval

完整示例见部分 InternNav 仓库中的 `rag4vln/scripts/eval/eval_rag4vln_streamvln.py`（本目录若未包含，可从上游拷贝）。要点：

- 增强后的 `--episode-json-gz` 直接作为 `data_path`。
- 在 StreamVLN 仓库根用 `torchrun … streamvln/streamvln_eval.py`，`PYTHONPATH` 同时包含 InternNav 根与 StreamVLN 根。
- 可将 StreamVLN 的 `result.json` 转成与 InternNav 一致的 `internnav_output/result.json` 便于对比。

### 6.4 新增一种指令增强策略

1. 在 `src/augment/` 新建类，继承 `InstructionAugmenter`，实现 `augment(instruction, evidence, …)`。
2. 在 `src/augment/__init__.py` 导出 `build_*_augmenter`。
3. 在 `eval_rag4vln_vln_augmented.py` 的 `_build_augmenter()` 与 argparse `choices` 中注册名称。
4. 在 `src/config.yaml` 的 `augment:` 下增加对应配置段（API、prompt 等）。

### 6.5 在 Python 中直接调用核心 API

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
print(result.instruction)  # 增强后的导航指令
```

将 `result.instruction` 写回你自己的 dataloader 或 json.gz，即可完成与任意 VLN 模型的对接。
