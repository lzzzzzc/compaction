"""Aggregate the eight key-compensation methods and paired Top/Random transitions."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PAIRS = {
    "raw_direct": ("key_analysis_top_raw_direct", "key_analysis_random_raw_direct"),
    "beta_direct": ("key_analysis_top_beta_direct", "key_analysis_random_beta_direct"),
    "zero_fitted": ("key_analysis_top_zero_fitted", "key_analysis_random_zero_fitted"),
    "beta_fitted": ("key_analysis_top_beta_fitted", "key_analysis_random_beta_fitted"),
}


def _safe_mean(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.mean(values) if values else None


def _question_key(article, question, position):
    question_id = (
        question.get("question_unique_id") or question.get("question_id")
        or question.get("id") or f"position_{position}"
    )
    return str(article.get("article_id", article.get("article_idx"))), str(question_id)


def _question_map(result):
    questions = result.get("qa_results", {}).get("results_per_question", [])
    return {
        _question_key(result, question, position): question
        for position, question in enumerate(questions)
    }


def _parseable(question):
    if "model_choice" in question:
        return question.get("model_choice") not in (None, "")
    return bool(question.get("answer", "").strip())


def _transition_counts(top_results, random_results):
    top_questions = {}
    random_questions = {}
    for result in top_results:
        top_questions.update(_question_map(result))
    for result in random_results:
        random_questions.update(_question_map(result))
    shared = sorted(set(top_questions) & set(random_questions))
    counts = defaultdict(int)
    for key in shared:
        top = top_questions[key]
        random = random_questions[key]
        top_correct = bool(top.get("is_correct", False))
        random_correct = bool(random.get("is_correct", False))
        if top_correct and random_correct:
            counts["both_correct"] += 1
        elif top_correct:
            counts["correct_to_wrong"] += 1
        elif random_correct:
            counts["wrong_to_correct"] += 1
        else:
            counts["both_wrong"] += 1
        top_parseable = _parseable(top)
        random_parseable = _parseable(random)
        counts[
            ("parseable" if top_parseable else "unanswered")
            + "_to_"
            + ("parseable" if random_parseable else "unanswered")
        ] += 1
    return {"paired_questions": len(shared), **dict(counts)}


def _method_row(stats):
    train = stats.get("overall_all_head_train_stats", {})
    test = stats.get("overall_all_head_test_stats", {})
    return {
        "accuracy": stats.get("overall_accuracy"),
        "parse_rate": stats.get("overall_parse_rate"),
        "perplexity": stats.get("overall_avg_perplexity"),
        "log_perplexity": stats.get("overall_avg_log_perplexity"),
        "train_reconstruction": train,
        "heldout_reconstruction": test,
    }


def aggregate(eval_dir):
    files = sorted(Path(eval_dir).rglob("*.json"))
    runs = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        methods = data.get("overall_stats", {})
        if not any(name.startswith("key_analysis_") for name in methods):
            continue
        runs.append((path, data))

    per_seed = {}
    transition_totals = {name: defaultdict(int) for name in PAIRS}
    method_values = defaultdict(lambda: defaultdict(list))
    mechanism = []
    for path, data in runs:
        eval_seed = data.get("config", {}).get("seed")
        key_seeds = {
            values.get("key_seed")
            for values in data.get("hyperparameters", {}).values()
            if isinstance(values, dict) and values.get("algorithm") == "key_selection_ablation"
        }
        key_seed = next(iter(key_seeds)) if len(key_seeds) == 1 else None
        seed_label = f"eval_seed={eval_seed},key_seed={key_seed}"
        per_seed.setdefault(seed_label, {})
        for method, stats in data.get("overall_stats", {}).items():
            if not method.startswith("key_analysis_"):
                continue
            row = _method_row(stats)
            per_seed[seed_label][method] = row
            for field in ("accuracy", "parse_rate", "perplexity", "log_perplexity"):
                if row[field] is not None:
                    method_values[method][field].append(row[field])
            if stats.get("key_selection_analysis_per_article"):
                mechanism.extend(stats["key_selection_analysis_per_article"])

        results_by_method = defaultdict(list)
        for result in data.get("results", []):
            results_by_method[result.get("method")].append(result)
        for pair_name, (top_method, random_method) in PAIRS.items():
            counts = _transition_counts(results_by_method[top_method], results_by_method[random_method])
            for field, value in counts.items():
                transition_totals[pair_name][field] += value

    multi_seed = {}
    for method, fields in method_values.items():
        multi_seed[method] = {}
        for field, values in fields.items():
            multi_seed[method][field] = {
                "mean": _safe_mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "num_runs": len(values),
            }

    return {
        "num_result_files": len(runs),
        "per_seed": per_seed,
        "multi_seed_summary": multi_seed,
        "paired_top_random_transitions": {
            name: dict(counts) for name, counts in transition_totals.items()
        },
        "key_selection_analysis_per_article": mechanism,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.eval_dir) / "key_compensation_analysis.json"
    result = aggregate(args.eval_dir)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
