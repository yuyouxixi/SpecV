#!/usr/bin/env python3
"""Compute per-question scores, track averages, overall averages, and rankings
for T2I model evaluation results."""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SCORES_DIR = BASE_DIR / "scores"

TRACKS = [
    "cross_domain",
    "element_replacement",
    "three_elements_integration",
    "multi_subject_composition",
]


def compute_question_score(filepath: Path) -> dict:
    """Return score info for a single question JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        items = json.load(f)
    total = len(items)
    correct = sum(
        1 for item in items if str(item["expected_answer"]) == str(item["answer"])
    )
    score = round(correct / total * 100, 2) if total > 0 else 0.0
    question_id = filepath.stem
    return {
        "question_id": question_id,
        "score": score,
        "correct": correct,
        "total": total,
    }


def process_model(model_dir: Path) -> dict:
    """Process all tracks for a single model. Returns track-level and overall data."""
    model_name = model_dir.name
    track_data = {}

    for track in TRACKS:
        track_dir = model_dir / track
        if not track_dir.is_dir():
            continue

        json_files = sorted(track_dir.glob("*.json"))
        questions = [compute_question_score(f) for f in json_files]
        track_avg = (
            round(sum(q["score"] for q in questions) / len(questions), 2)
            if questions
            else 0.0
        )
        track_data[track] = {
            "track": track,
            "track_average": track_avg,
            "questions": questions,
        }

    track_avgs = {t: td["track_average"] for t, td in track_data.items()}
    overall_avg = (
        round(sum(track_avgs.values()) / len(track_avgs), 2) if track_avgs else 0.0
    )

    return {
        "model": model_name,
        "overall_average": overall_avg,
        "tracks": track_avgs,
        "track_detail": track_data,
    }


def write_model_scores(model_result: dict) -> None:
    """Write per-track detail files and summary.json for one model."""
    model_name = model_result["model"]
    model_scores_dir = SCORES_DIR / model_name
    model_scores_dir.mkdir(parents=True, exist_ok=True)

    for track, detail in model_result["track_detail"].items():
        out_path = model_scores_dir / f"{track}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)

    summary = {
        "model": model_name,
        "overall_average": model_result["overall_average"],
        "tracks": model_result["tracks"],
    }
    with open(model_scores_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def write_rankings(all_results: list[dict]) -> None:
    """Write overall ranking and per-track ranking files."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    sorted_overall = sorted(all_results, key=lambda r: r["overall_average"], reverse=True)
    overall_ranking = [
        {"rank": i + 1, "model": r["model"], "score": r["overall_average"]}
        for i, r in enumerate(sorted_overall)
    ]
    with open(SCORES_DIR / "overall_ranking.json", "w", encoding="utf-8") as f:
        json.dump(overall_ranking, f, indent=2, ensure_ascii=False)

    for track in TRACKS:
        sorted_track = sorted(
            all_results,
            key=lambda r, t=track: r["tracks"].get(t, 0.0),
            reverse=True,
        )
        track_ranking = [
            {"rank": i + 1, "model": r["model"], "score": r["tracks"].get(track, 0.0)}
            for i, r in enumerate(sorted_track)
        ]
        with open(SCORES_DIR / f"{track}_ranking.json", "w", encoding="utf-8") as f:
            json.dump(track_ranking, f, indent=2, ensure_ascii=False)


def main():
    model_dirs = sorted(
        [d for d in RESULTS_DIR.iterdir() if d.is_dir()], key=lambda d: d.name
    )
    print(f"Found {len(model_dirs)} models: {[d.name for d in model_dirs]}")

    all_results = []
    for model_dir in model_dirs:
        result = process_model(model_dir)
        write_model_scores(result)
        all_results.append(result)
        print(f"  {result['model']}: overall={result['overall_average']}, tracks={result['tracks']}")

    write_rankings(all_results)

    print(f"\nDone. Scores written to {SCORES_DIR}")
    print("\n=== Overall Ranking ===")
    sorted_overall = sorted(all_results, key=lambda r: r["overall_average"], reverse=True)
    for i, r in enumerate(sorted_overall, 1):
        print(f"  {i}. {r['model']}: {r['overall_average']}")


if __name__ == "__main__":
    main()
