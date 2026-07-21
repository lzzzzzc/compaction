"""JSON-safe metrics for comparing two compressed-key supports."""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _finite(value: torch.Tensor) -> Optional[float]:
    result = float(value.item())
    return result if math.isfinite(result) else None


def _summary(values: torch.Tensor) -> Dict:
    values = values.float()
    if values.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": _finite(values.mean()),
        "std": _finite(values.std(unbiased=False)),
        "min": _finite(values.min()),
        "max": _finite(values.max()),
    }


def _internal_geometry(keys_normalized: torch.Tensor) -> Dict:
    m = keys_normalized.shape[0]
    if m <= 1:
        return {
            "pairwise_cosine": _summary(keys_normalized.new_empty(0)),
            "nearest_neighbor_redundancy_mean": None,
            "nearest_neighbor_redundancy_max": None,
        }
    cosine = keys_normalized @ keys_normalized.T
    mask = ~torch.eye(m, dtype=torch.bool, device=cosine.device)
    pairwise = cosine[mask]
    nearest = cosine.masked_fill(~mask, float("-inf")).max(dim=1).values
    return {
        "pairwise_cosine": _summary(pairwise),
        "nearest_neighbor_redundancy_mean": _finite(nearest.mean()),
        "nearest_neighbor_redundancy_max": _finite(nearest.max()),
    }


def _linear_cka(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> Optional[float]:
    """Centered linear CKA through query Gram matrices; slot columns need not align."""
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    cross = left.T @ right
    numerator = (cross * cross).sum()
    left_norm = torch.linalg.matrix_norm(left.T @ left, ord="fro")
    right_norm = torch.linalg.matrix_norm(right.T @ right, ord="fro")
    denominator = left_norm * right_norm
    if denominator <= eps:
        return None
    return _finite(numerator / denominator)


def _routing_spectrum(routing: torch.Tensor, topk: int) -> Dict:
    singular = torch.linalg.svdvals(routing.float())
    if singular.numel() == 0:
        return {
            "singular_values": [], "effective_rank": 0.0, "stable_rank": 0.0,
            "condition_number": None, "rank_deficient": True,
            "energy_rank_90": 0, "energy_rank_95": 0, "energy_rank_99": 0,
        }
    singular_sum = singular.sum()
    probabilities = singular / singular_sum.clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    squared = singular.square()
    stable_rank = squared.sum() / squared[0].clamp_min(1e-12)
    tolerance = max(routing.shape) * torch.finfo(torch.float32).eps * singular[0]
    rank_deficient = bool((singular[-1] <= tolerance).item())
    condition = None if rank_deficient else _finite(singular[0] / singular[-1])
    cumulative = squared.cumsum(dim=0) / squared.sum().clamp_min(1e-12)

    def energy_rank(threshold: float) -> int:
        found = int(torch.searchsorted(
            cumulative, torch.tensor(threshold, device=cumulative.device)
        ).item() + 1)
        return min(found, int(singular.numel()))

    return {
        "singular_values": [float(x) for x in singular[:topk].cpu().tolist()],
        "effective_rank": _finite(effective_rank),
        "stable_rank": _finite(stable_rank),
        "condition_number": condition,
        "rank_deficient": rank_deficient,
        "energy_rank_90": energy_rank(0.90),
        "energy_rank_95": energy_rank(0.95),
        "energy_rank_99": energy_rank(0.99),
    }


def _slot_usage(routing: torch.Tensor, dead_slot_threshold: float) -> Dict:
    usage = routing.float().mean(dim=0)
    entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
    max_entropy = math.log(max(1, usage.numel()))
    return {
        **_summary(usage),
        "entropy": _finite(entropy),
        "normalized_entropy": _finite(entropy / max_entropy) if max_entropy > 0 else 0.0,
        "dead_slot_threshold": float(dead_slot_threshold),
        "dead_slot_count": int((usage < dead_slot_threshold).sum().item()),
    }


def _routing_pair(left: torch.Tensor, right: torch.Tensor, spectrum_topk: int,
                  dead_slot_threshold: float) -> Dict:
    left = left.float()
    right = right.float()
    gram_left = left @ left.T
    gram_right = right @ right.T
    gram_error = (
        torch.linalg.matrix_norm(gram_left - gram_right, ord="fro")
        / torch.linalg.matrix_norm(gram_left, ord="fro").clamp_min(1e-12)
    )
    return {
        "routing_gram_error": _finite(gram_error),
        "linear_cka": _linear_cka(left, right),
        "top": {
            "spectrum": _routing_spectrum(left, spectrum_topk),
            "slot_usage": _slot_usage(left, dead_slot_threshold),
        },
        "random": {
            "spectrum": _routing_spectrum(right, spectrum_topk),
            "slot_usage": _slot_usage(right, dead_slot_threshold),
        },
    }


def _hungarian_similarity(similarity: torch.Tensor, max_slots: int) -> Dict:
    m = similarity.shape[0]
    if m > max_slots:
        return {"value": None, "skipped_reason": f"m={m} exceeds hungarian_max_slots={max_slots}"}
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return {"value": None, "skipped_reason": "scipy is not installed"}
    rows, cols = linear_sum_assignment((-similarity.float().cpu().numpy()))
    matched = similarity[torch.as_tensor(rows, device=similarity.device),
                         torch.as_tensor(cols, device=similarity.device)]
    return {"value": _finite(matched.mean()), "skipped_reason": None}


def analyze_key_selections(
    top_keys: torch.Tensor,
    random_keys: torch.Tensor,
    top_indices: torch.Tensor,
    random_indices: torch.Tensor,
    queries: torch.Tensor,
    top_beta: torch.Tensor,
    random_beta: torch.Tensor,
    candidate_count: int,
    dead_slot_threshold: float = 1e-4,
    spectrum_topk: int = 20,
    hungarian_max_slots: int = 512,
) -> Dict:
    """Compare supports, key geometry, and raw/fitted routing on shared queries."""
    top_indices = top_indices.long()
    random_indices = random_indices.long()
    m = int(top_indices.numel())
    if m != int(random_indices.numel()):
        raise ValueError("Top and Random supports must have the same budget")
    if m > candidate_count or top_indices.unique().numel() != m or random_indices.unique().numel() != m:
        raise ValueError("Supports must be unique subsets of the shared candidate range")
    if m and (top_indices.min() < 0 or random_indices.min() < 0 or
              top_indices.max() >= candidate_count or random_indices.max() >= candidate_count):
        raise ValueError("Support index falls outside the shared candidate range")

    intersection = len(set(top_indices.cpu().tolist()) & set(random_indices.cpu().tolist()))
    union = 2 * m - intersection
    expected_ratio = (m / candidate_count) if candidate_count else 0.0
    overlap_ratio = (intersection / m) if m else 0.0

    top_norms = top_keys.float().norm(dim=1)
    random_norms = random_keys.float().norm(dim=1)
    top_normalized = F.normalize(top_keys.float(), dim=1, eps=1e-12)
    random_normalized = F.normalize(random_keys.float(), dim=1, eps=1e-12)
    similarity = top_normalized @ random_normalized.T
    top_to_random = similarity.max(dim=1).values if m else similarity.new_empty(0)
    random_to_top = similarity.max(dim=0).values if m else similarity.new_empty(0)

    d = top_keys.shape[1]
    scale = d ** -0.5
    top_scores = (queries @ top_keys.T).float() * scale
    random_scores = (queries @ random_keys.T).float() * scale
    top_raw = torch.softmax(top_scores, dim=1)
    random_raw = torch.softmax(random_scores, dim=1)
    top_fit = torch.softmax(top_scores + top_beta.float(), dim=1)
    random_fit = torch.softmax(random_scores + random_beta.float(), dim=1)

    return {
        "candidate_count": int(candidate_count),
        "budget": m,
        "overlap": {
            "intersection_size": int(intersection),
            "overlap_ratio": float(overlap_ratio),
            "jaccard": float(intersection / union) if union else 1.0,
            "expected_overlap_ratio": float(expected_ratio),
            "expected_intersection_size": float(m * m / candidate_count) if candidate_count else 0.0,
            "normalized_overlap": float(overlap_ratio / expected_ratio) if expected_ratio else None,
        },
        "geometry": {
            "top_to_random_nearest": _summary(top_to_random),
            "random_to_top_nearest": _summary(random_to_top),
            "bidirectional_key_similarity": _finite(
                0.5 * (top_to_random.mean() + random_to_top.mean())) if m else None,
            "hungarian_key_similarity": _hungarian_similarity(similarity, hungarian_max_slots),
            "top_key_norm": _summary(top_norms),
            "random_key_norm": _summary(random_norms),
            "top_internal": _internal_geometry(top_normalized),
            "random_internal": _internal_geometry(random_normalized),
        },
        "routing": {
            "raw": _routing_pair(top_raw, random_raw, spectrum_topk, dead_slot_threshold),
            "fitted_bias": _routing_pair(top_fit, random_fit, spectrum_topk, dead_slot_threshold),
        },
    }
