# SpecV

**SpecV: Specification Verification for Robust Unified Multimodal Evaluation** (ECCV 2026)

SpecV is a benchmark for **unified multimodal models (UMMs)** that evaluates visual understanding,
image generation, editing, and interleaved image–text generation across **6 tracks** with the
*Specification Verification Protocol (SVP)*. Rather than collapsing an output into a single opaque
score, SVP decomposes each prompt into atomic, binary **specifications** (yes/no checklist items) and
has a judge model verify each one against the model's output. Scoring becomes transparent,
decomposable, and stable across different judge models, so model rankings stay consistent even when
the judge changes.

SpecV-Bench spans **1,200 instances**, **28 sub-tracks**, and **17,604 specifications** (≈14.7 per instance).

<p align="center">
  <a href="https://github.com/yuyouxixi/SpecV"><img src="https://img.shields.io/badge/GitHub-SpecV-181717?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/vortex778/SpecV"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow" alt="Hugging Face Dataset"></a>
  <img src="https://img.shields.io/badge/Venue-ECCV%202026-4b44ce" alt="ECCV 2026">
</p>

- 📄 Paper: **SpecV: Specification Verification for Robust Unified Multimodal Evaluation** (ECCV 2026) — *arXiv link coming soon*
- 🤗 Dataset (benchmark images + released model outputs): <https://huggingface.co/datasets/vortex778/SpecV>

---

## Contents
- [The 6 tracks](#the-6-tracks)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Data formats](#data-formats)
- [Running the evaluation](#running-the-evaluation)
- [Evaluated models](#evaluated-models)
- [Leaderboard](#leaderboard)
- [Citation](#citation) · [Terms of use](#terms-of-use)

---

## The 6 tracks

| Track | Directory | Description | #Tasks | Subtasks (dimensions) |
|------|-----------|-------------|:------:|-----------------------|
| Text-to-Image | `t2i` | Generate an image from a detailed prompt | 200 | composition, structure, style, text_rendering, world_knowledge_and_reasoning |
| Image Editing | `edit` | Edit a source image per an instruction | 200 | attribute, background_and_scene, local_obj, reasoning_driven, spatial_and_layout, text_centric |
| Many-to-One | `many2one` | Compose one image from 2–3 reference images | 200 | cross_domain, element_replacement, multi_subject_composition, three_elements_integration |
| Interleaved Generation | `interleave` | Produce interleaved text + images | 200 | interleaved_completion, real_world_assistant, reasoning_interleaved, structured_tutorial |
| Thinking with Images | `think_with_img` | Reason by generating intermediate images | 200 | math, physics, jigsaw, maze |
| Understanding | `understanding` | Answer questions about an image | 200 | charxiv, countbench, hallusionbench, mmbench, mmmu |

Each task comes with a checklist of weighted yes/no specifications under `eval_checklist/<track>/`.

---

## Repository layout

```
SpecV/
├── eval_data/                 # benchmark tasks: <track>/*.jsonl   (images fetched from HF)
├── eval_checklist/            # per-task checklists: <track>/<subtask>/<id>.json
├── eval/                      # evaluation code (SVP checklist scoring)
│   └── <track>/
│       ├── run_eval.py             # SVP (checklist) verification with a judge model
│       ├── score_and_rank.py       # aggregate per-item judgements into scores + rankings
│       └── eval_prompt.txt         # checklist judge prompt
├── model_outputs/             # released model outputs (fetched from HF)
├── scripts/download_data.py   # download images + model outputs from the HF dataset
├── docs/HF_UPLOAD.md          # (maintainers) how the HF dataset is published
├── requirements.txt · .env.example
```

`eval_data/<track>/*.jsonl` and `eval_checklist/` ship in this repo. The benchmark **images** and
the **model outputs** are large and hosted on Hugging Face — see [Data](#data).

---

## Installation

```bash
git clone https://github.com/yuyouxixi/SpecV.git
cd SpecV
pip install -r requirements.txt

# fetch benchmark images + released model outputs from the HF dataset
python scripts/download_data.py            # or: --what images   /   --what outputs

# configure your judge (API key + endpoint + model id)
cp .env.example .env && edit .env          # sets DASHSCOPE_API_KEY, JUDGE_API_URL, JUDGE_MODEL
```

SpecV was evaluated with **Gemini 3 Flash** as the judge. The judge endpoint and model id are yours
to configure: set `JUDGE_API_URL` (judge API endpoint), `JUDGE_MODEL` (the model id your provider
exposes), and `DASHSCOPE_API_KEY` (API key). `--judge-model` overrides `JUDGE_MODEL` for a single run.

---

## Data

The lean repo + Hugging Face split:

| Component | Where | Size |
|-----------|-------|------|
| Task definitions (`eval_data/<track>/*.jsonl`) | this repo | small |
| Checklists (`eval_checklist/`) | this repo | ~6 MB |
| Evaluation code (`eval/`) | this repo | small |
| Benchmark **input images** | 🤗 HF dataset → `eval_data/<track>/images/` | ~1.6 GB |
| Released **model outputs** | 🤗 HF dataset → `model_outputs/<track>/<model>/` | ~16 GB |

`scripts/download_data.py` downloads the images and model outputs into the repo root so the paths
line up with the code. Use `--what images` if you only need the benchmark to evaluate your own
model, or `--tasks t2i edit` to restrict to specific tracks.

> The released model outputs are model **generations only**. Computed evaluation scores are reported
> in the paper.

---

## Data formats

**Task JSONL** — one JSON object per line. Fields vary by track:

| Field | Tracks | Meaning |
|-------|--------|---------|
| `id` | all | unique task id (e.g. `attr_001`, `math_001`) |
| `question` | all | prompt / instruction / question (may contain `[IMAGE1]`, `[IMAGE2]`, … placeholders) |
| `dimension` | t2i, edit, many2one, interleave | subtask name |
| `question_img` | edit, many2one, interleave\*, think_with_img, understanding | list of input-image paths, relative to the track dir, e.g. `images/attribute/attr_001.png` |
| `answer` | think_with_img, understanding | reference answer |
| `answer_img` | think_with_img | reference-answer image path(s) |

\* `interleave` only has input images for the `interleaved_completion` subtask. `t2i` has no input images.

```json
{"id": "attr_001", "dimension": "attribute",
 "question": "Transform this red car into a brushed stainless-steel finish.",
 "question_img": ["images/attribute/attr_001.png"]}
```

**Checklist JSON** — a list of weighted specifications per task:

```json
[
  {"id": 1, "question": "Is the car body now predominantly silver?",
   "category": "Edit Execution", "subcategory": "Color",
   "expected_answer": "Yes", "score": 9.5}
]
```

**Expected model-output layout** (what `run_eval.py` reads, under `model_outputs/<track>/<model>/`):

| Track(s) | Format | Per-task output |
|----------|--------|-----------------|
| t2i, edit, many2one | single image | `<id>.png` (or `.jpg`) |
| interleave, think_with_img | interleaved | `<id>/response.json` + `<id>/1.png, 2.png, …` |
| understanding | text JSON | `<id>.json` = `{"id", "response", "error"}` |

`response.json` for interleaved tracks:
```json
{"content": [{"type": "text", "text": "..."},
             {"type": "image", "image": "1.png"},
             {"type": "text", "text": "..."}]}
```

---

## Running the evaluation

SpecV scores a model with the **Specification Verification Protocol (SVP)**: for each task the judge
receives the model's output, the prompt, and the checklist, and answers yes/no for every
specification.

```bash
# configure your judge (SpecV was evaluated with Gemini 3 Flash)
export DASHSCOPE_API_KEY=<your-key>
export JUDGE_API_URL=<your-judge-api-endpoint>
export JUDGE_MODEL=<your-judge-model-id>

# evaluate one model on one track
python eval/t2i/run_eval.py --model-name nano_banana --num-processes 50

# several models at once
python eval/edit/run_eval.py --model-name nano_banana gpt_image_1

# aggregate scores + rankings for that track
python eval/t2i/score_and_rank.py
```

`run_eval.py` writes per-item judgements to `eval/<track>/results/<model>/<dimension>/<id>.json`.
`score_and_rank.py` then computes, for each model:
- **per-task score** = fraction of checklist items answered as the `expected_answer` (×100),
- **per-dimension** average, and **overall** average, plus **rankings**, written to
  `eval/<track>/scores/`.

The judge model id and endpoint are user-supplied — set `JUDGE_MODEL` and `JUDGE_API_URL` (and
`DASHSCOPE_API_KEY`); `--judge-model` overrides `JUDGE_MODEL` for a single run. Runs are resumable
(finished items are skipped) and parallel (`--num-processes`); use `--limit N` for a quick test.

To evaluate across all six tracks, loop over them:

```bash
for track in understanding t2i edit many2one interleave think_with_img; do
  python eval/$track/run_eval.py --model-name nano_banana --num-processes 50
  python eval/$track/score_and_rank.py
done
```

---

## Evaluated models

Released model outputs, by track (paper name → directory name):

| Paper name | Directory | t2i | edit | m2o | interleave | think_w_img | understanding |
|------------|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| Nano Banana | `nano_banana` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Nano Banana Pro | `nano_banana_pro` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Nano Banana 2 | `nano_banana_2` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPT Image 1 | `gpt_image_1` | ✅ | ✅ | ✅ | | | |
| GPT Image 1.5 | `gpt_image_1.5` | ✅ | ✅ | ✅ | | | |
| Seedream 5.0 | `seedream_5.0` | ✅ | ✅ | ✅ | | | |
| GLM-Image | `glm_image` | ✅ | | | | | |
| Z-Image-Turbo | `z_image_turbo` | ✅ | | | | | |
| Qwen-Image | `qwen_image` | ✅ | ✅ | ✅ | | | |
| Qwen-Image-2 | `qwen_image_2` | ✅ | ✅ | ✅ | | | |
| Bagel (w/o CoT) | `bagel` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bagel (w/ CoT) | `bagel_think` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Emu 3.5 | `emu3.5` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| BLIP3o-NeXT | `blip3o_next` | ✅ | | | | | |
| Gemini 3 Flash | `gemini_3_flash` | | | | | | ✅ |
| Gemini 3 Pro | `gemini_3_pro` | | | | | | ✅ |
| Gemini 3.1 Pro | `gemini_3.1_pro` | | | | | | ✅ |
| GPT-5.1 | `gpt_5.1` | | | | | | ✅ |
| GPT-5.2 | `gpt_5.2` | | | | | | ✅ |
| Claude Opus 4.5 | `claude_opus_4.5` | | | | | | ✅ |
| Qwen3.5-Plus | `qwen3.5_plus` | | | | | | ✅ |

---

## Leaderboard

Scores are the **SVP Score** — the percentage of checklist specifications an output satisfies
(**higher is better**). All numbers are from the paper (judge: Gemini 3 Flash). The main board ranks
the **unified multimodal models** that cover all six tracks by their **Average**; specialist
generation/editing models and understanding-only VLMs appear in the per-track tables below.

| Rank | Model | Underst. | T2I | Editing | Many→One | Interleave | Think w/ Img | **Average** |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 🥇 | **Nano Banana 2** | 83.25 | 90.45 | 90.34 | 76.33 | 80.27 | 69.97 | **81.77** |
| 🥈 | **Nano Banana Pro** | 83.25 | 90.23 | 88.42 | 74.31 | 74.63 | 61.81 | **78.78** |
| 🥉 | Nano Banana | 78.21 | 68.03 | 74.95 | 68.74 | 82.04 | 51.00 | **70.50** |
| 4 | Qwen-Image-2 | 71.00 | 75.28 | 73.30 | 67.73 | 74.56 | 32.47 | **65.72** |
| 5 | Emu 3.5 | 45.99 | 67.14 | 64.53 | 66.53 | 62.23 | 14.18 | **53.43** |
| 6 | Qwen-Image | 57.00 | 64.60 | 69.16 | 60.82 | 34.03 | 19.84 | **50.91** |
| 7 | Bagel (w/ CoT) | 69.00 | 26.39 | 56.88 | 45.96 | 45.06 | 18.34 | **43.61** |
| 8 | Bagel (w/o CoT) | 63.75 | 28.09 | 54.45 | 47.35 | 35.90 | 16.81 | **41.06** |

<details>
<summary><b>📊 Per-track leaderboards</b> — click to expand (includes specialist generation/editing models and understanding-only VLMs)</summary>

#### Multimodal Understanding
<sub>ChartR. = Chart Reasoning · Count = Object Counting · Anti-H. = Anti-Hallucination · MMQA = Multimodal QA · MDK = Multidisciplinary Knowledge</sub>

| Rank | Model | ChartR. | Count | Anti-H. | MMQA | MDK | **Avg** |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Gemini 3.1 Pro | 87.50 | 90.00 | 85.00 | 97.50 | 85.00 | **89.00** |
| 2 | Gemini 3 Pro | 78.75 | 95.00 | 85.00 | 97.50 | 85.00 | **88.25** |
| 3 | GPT 5.2 | 94.87 | 85.00 | 80.00 | 95.00 | 81.58 | **87.29** |
| 4 | Qwen 3.5 Plus | 78.21 | 90.00 | 87.50 | 90.00 | 80.00 | **85.14** |
| 5 | Nano Banana Pro | 73.75 | 95.00 | 80.00 | 92.50 | 75.00 | **83.25** |
| 6 | Nano Banana 2 | 71.25 | 92.50 | 82.50 | 97.50 | 72.50 | **83.25** |
| 7 | Claude Opus 4.5 | 70.00 | 86.25 | 75.00 | 95.00 | 65.00 | **78.25** |
| 8 | Nano Banana | 60.00 | 91.03 | 75.00 | 95.00 | 70.00 | **78.21** |
| 9 | Qwen-Image-2 | 58.75 | 86.25 | 72.50 | 90.00 | 47.50 | **71.00** |
| 10 | Bagel (w/ CoT) | 55.00 | 92.50 | 75.00 | 87.50 | 35.00 | **69.00** |
| 11 | Bagel (w/o CoT) | 46.25 | 82.50 | 67.50 | 81.25 | 41.25 | **63.75** |
| 12 | Qwen-Image | 31.25 | 72.50 | 65.00 | 87.50 | 28.75 | **57.00** |
| 13 | Emu 3.5 | 16.25 | 73.75 | 55.00 | 47.44 | 37.50 | **45.99** |

#### Text-to-Image
<sub>Comp. = Compositional · Text = Text Rendering · Know. = World Knowledge · Struct. = Structural · Style = Style Adherence</sub>

| Rank | Model | Comp. | Text | Know. | Struct. | Style | **Avg** |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Nano Banana 2 | 89.34 | 83.19 | 91.84 | 91.31 | 96.58 | **90.45** |
| 2 | Nano Banana Pro | 89.83 | 81.30 | 90.73 | 93.84 | 95.44 | **90.23** |
| 3 | Seedream 5.0 | 86.53 | 58.03 | 87.95 | 83.35 | 90.64 | **81.30** |
| 4 | GPT Image 1.5 | 83.61 | 68.75 | 84.99 | 75.33 | 88.52 | **80.24** |
| 5 | Qwen-Image-2 | 83.99 | 67.07 | 68.98 | 68.39 | 87.97 | **75.28** |
| 6 | Nano Banana | 78.04 | 36.58 | 78.59 | 61.00 | 85.96 | **68.03** |
| 7 | Emu 3.5 | 80.25 | 51.19 | 63.20 | 55.16 | 85.92 | **67.14** |
| 8 | GPT Image 1 | 70.84 | 43.75 | 79.47 | 59.03 | 80.85 | **66.79** |
| 9 | Qwen-Image | 79.04 | 55.18 | 60.74 | 47.74 | 80.29 | **64.60** |
| 10 | Z-Image-turbo | 70.47 | 43.10 | 52.23 | 39.37 | 65.55 | **54.14** |
| 11 | GLM-Image | 46.33 | 26.39 | 61.05 | 46.13 | 47.03 | **45.49** |
| 12 | Bagel (w/o CoT) | 40.45 | 15.44 | 43.15 | 13.91 | 27.48 | **28.09** |
| 13 | Bagel (w/ CoT) | 29.66 | 12.29 | 46.98 | 16.37 | 26.63 | **26.39** |
| 14 | BLIP3o-NeXT | 33.34 | 8.86 | 40.27 | 8.71 | 36.80 | **25.60** |

#### Image Editing
<sub>Attr. = Attribute · Scene = Scene & Background · Local = Local Object · Reas. = Reasoning-driven · Spatial = Spatial & Layout · Text = Text Editing</sub>

| Rank | Model | Attr. | Scene | Local | Reas. | Spatial | Text | **Avg** |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Nano Banana 2 | 96.04 | 96.88 | 90.90 | 86.96 | 89.31 | 81.97 | **90.34** |
| 2 | Nano Banana Pro | 93.62 | 95.19 | 89.86 | 89.60 | 82.10 | 80.12 | **88.42** |
| 3 | Seedream 5.0 | 96.92 | 92.37 | 94.33 | 75.05 | 90.54 | 61.65 | **85.14** |
| 4 | GPT Image 1.5 | 83.39 | 87.19 | 85.13 | 70.70 | 89.71 | 70.69 | **81.14** |
| 5 | Nano Banana | 93.28 | 83.28 | 85.53 | 70.70 | 89.71 | 70.69 | **74.95** |
| 6 | Qwen-Image-2 | 94.43 | 93.08 | 90.70 | 43.67 | 73.74 | 44.19 | **73.30** |
| 7 | Qwen-Image | 88.72 | 84.08 | 91.07 | 38.53 | 76.03 | 36.53 | **69.16** |
| 8 | GPT Image 1 | 78.59 | 82.96 | 74.16 | 54.45 | 82.37 | 39.34 | **68.64** |
| 9 | Emu 3.5 | 79.38 | 86.07 | 77.95 | 47.42 | 69.92 | 26.43 | **64.53** |
| 10 | Bagel (w/ CoT) | 77.92 | 70.99 | 53.29 | 42.28 | 56.27 | 40.50 | **56.88** |
| 11 | Bagel (w/o CoT) | 75.95 | 71.89 | 65.10 | 41.27 | 48.89 | 23.61 | **54.45** |

#### Many-to-One Generation
<sub>Cross = Cross-domain · Repl. = Element Replacement · 3-Integ. = Three-element Integration · Multi = Multi-subject Composition</sub>

| Rank | Model | Cross | Repl. | 3-Integ. | Multi | **Avg** |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | GPT Image 1.5 | 81.16 | 68.79 | 81.45 | 79.14 | **77.64** |
| 2 | Nano Banana 2 | 76.67 | 66.42 | 86.49 | 75.73 | **76.33** |
| 3 | Nano Banana Pro | 72.38 | 66.80 | 82.21 | 75.85 | **74.31** |
| 4 | GPT Image 1 | 76.60 | 63.34 | 78.28 | 74.71 | **73.23** |
| 5 | Seedream 5.0 | 74.94 | 64.28 | 77.12 | 76.07 | **73.10** |
| 6 | Nano Banana | 67.35 | 67.06 | 71.66 | 68.90 | **68.74** |
| 7 | Qwen-Image-2 | 69.18 | 65.68 | 62.38 | 73.67 | **67.73** |
| 8 | Emu 3.5 | 71.03 | 56.74 | 71.17 | 67.19 | **66.53** |
| 9 | Qwen-Image | 69.87 | 56.91 | 63.80 | 52.69 | **60.82** |
| 10 | Bagel (w/o CoT) | 50.64 | 33.15 | 49.95 | 55.67 | **47.35** |
| 11 | Bagel (w/ CoT) | 50.65 | 35.10 | 49.76 | 48.34 | **45.96** |

#### Interleaved Generation
<sub>Compl. = Interleaved Completion · Tutor. = Structural Tutorial · Real. = Real-world Assistant · Reas. = Reasoning-based Generation</sub>

| Rank | Model | Compl. | Tutor. | Real. | Reas. | **Avg** |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | Nano Banana | 87.65 | 77.47 | 87.99 | 75.05 | **82.04** |
| 2 | Nano Banana 2 | 76.04 | 78.18 | 85.33 | 81.52 | **80.27** |
| 3 | Nano Banana Pro | 84.73 | 67.39 | 77.98 | 68.42 | **74.63** |
| 4 | Qwen-Image-2 | 81.11 | 67.40 | 80.80 | 68.92 | **74.56** |
| 5 | Emu 3.5 | 67.22 | 64.15 | 69.45 | 48.11 | **62.23** |
| 6 | Bagel (w/ CoT) | 52.84 | 45.90 | 41.34 | 40.16 | **45.06** |
| 7 | Bagel (w/o CoT) | 33.82 | 27.69 | 51.75 | 30.35 | **35.90** |
| 8 | Qwen-Image | 41.41 | 25.85 | 42.16 | 26.70 | **34.03** |

#### Thinking with Images
<sub>Math · Phys. = Physics · Maze · Jigsaw</sub>

| Rank | Model | Math | Phys. | Maze | Jigsaw | **Avg** |
|:--:|:--|:--:|:--:|:--:|:--:|:--:|
| 1 | Nano Banana 2 | 87.32 | 93.28 | 46.74 | 52.53 | **69.97** |
| 2 | Nano Banana Pro | 78.66 | 78.96 | 49.09 | 40.55 | **61.81** |
| 3 | Nano Banana | 67.11 | 77.09 | 25.31 | 34.16 | **51.00** |
| 4 | Qwen-Image-2 | 42.90 | 46.50 | 13.84 | 26.65 | **32.47** |
| 5 | Qwen-Image | 27.14 | 20.34 | 11.77 | 20.09 | **19.84** |
| 6 | Bagel (w/ CoT) | 25.40 | 26.31 | 8.34 | 13.32 | **18.34** |
| 7 | Bagel (w/o CoT) | 23.42 | 26.42 | 13.55 | 3.85 | **16.81** |
| 8 | Emu 3.5 | 15.28 | 20.25 | 15.61 | 5.59 | **14.18** |

</details>

---

## Citation

If you use SpecV, please cite:

```bibtex
@inproceedings{yu2026specv,
  title     = {SpecV: Specification Verification for Robust Unified Multimodal Evaluation},
  author    = {Yu, Weihao and Fang, Rongyao and Cai, Yuxuan and Huang, Linjiang and Yang, Yuhuan and Zhuang, Xianwei and Lin, Junyang and Yuan, Yixuan and Bai, Shuai},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Terms of use

Benchmark images and model outputs are provided for research use; please respect the licenses/terms
of the underlying source datasets (CharXiv, CountBench, HallusionBench, MMBench, MMMU) and the
evaluated models.
