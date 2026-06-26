#!/usr/bin/env python3
"""Download SpecV benchmark images and model outputs from the Hugging Face Hub.

The GitHub repo is intentionally lean: it ships the task definitions
(``eval_data/<task>/*.jsonl``) and the evaluation checklists (``eval_checklist/``).
The large binaries — the benchmark *input images* and the released *model outputs* —
live in a Hugging Face dataset and are downloaded into the repo root so the evaluation
scripts find them at the expected locations:

    eval_data/<task>/images/...                 # benchmark input images
    eval_data/think_with_img/answer_images/...  # reference-answer images
    model_outputs/<task>/<model>/...            # released model outputs

Examples
--------
    # everything (images + all released model outputs)
    python scripts/download_data.py

    # only the benchmark input images (enough to run your own models)
    python scripts/download_data.py --what images

    # only the released model outputs
    python scripts/download_data.py --what outputs

    # restrict to a couple of tracks
    python scripts/download_data.py --tasks t2i edit
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# NOTE: set this to the actual Hugging Face dataset repo id before release,
# or pass --repo-id on the command line.
DEFAULT_REPO_ID = "yuyouxixi/SpecV"

TASKS = ["t2i", "edit", "many2one", "interleave", "think_with_img", "understanding"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download SpecV images + model outputs from the Hugging Face Hub."
    )
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"HF dataset repo id, e.g. user/name (default: {DEFAULT_REPO_ID})")
    ap.add_argument("--revision", default="main", help="dataset git revision/tag (default: main)")
    ap.add_argument("--what", choices=["all", "images", "outputs"], default="all",
                    help="download benchmark images, model outputs, or both (default: all)")
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS,
                    help="restrict to specific tracks (default: all 6)")
    ap.add_argument("--local-dir", default=str(REPO_ROOT),
                    help="destination directory (default: the repo root, so paths line up)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    patterns = []
    for t in args.tasks:
        if args.what in ("all", "images"):
            patterns.append(f"eval_data/{t}/**")
        if args.what in ("all", "outputs"):
            patterns.append(f"model_outputs/{t}/**")

    print(f"Downloading from dataset '{args.repo_id}' (revision: {args.revision})")
    print("Patterns:")
    for p in patterns:
        print("  ", p)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=patterns,
    )

    print(f"\nDone. Files placed under: {args.local_dir}")
    print("Next, e.g.:")
    print("  export DASHSCOPE_API_KEY=sk-...")
    print("  python eval/t2i/run_eval.py --model-name nano_banana --limit 5 --num-processes 2")


if __name__ == "__main__":
    main()
