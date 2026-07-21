param(
    [string]$ModelName = "Qwen/Qwen3-4B",
    [string]$QueryConfig = "evaluation/configs/query_generation/repeat.py",
    [string]$BudgetPath = "head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json",
    [int]$StartArticle = 0,
    [int]$NArticles = 1,
    [int]$Seed = 0,
    [int]$KeySeed = 0,
    [switch]$Diagnostic
)

$common = @(
    "-m", "evaluation.run_qa_evaluation",
    "--methods", "all",
    "--algorithm-config", "evaluation/configs/algorithms/key-compensation-analysis.py",
    "--query-config", $QueryConfig,
    "--model-name", $ModelName,
    "--precomputed-budget-path", $BudgetPath,
    "--max-ratio-per-head", "0.75",
    "--start-article", $StartArticle,
    "--n-articles", $NArticles,
    "--seed", $Seed,
    "--key-seed", $KeySeed,
    "--max-new-tokens", "2048",
    "--batch-size", "1"
)

if ($Diagnostic) {
    $common += @("--compute-stats", "1", "--compute-perplexity", "1", "--verbose-logging", "1")
} else {
    $common += @("--compute-stats", "0", "--compute-perplexity", "0", "--verbose-logging", "0")
}

python @common
