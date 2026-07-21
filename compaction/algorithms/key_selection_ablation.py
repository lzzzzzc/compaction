"""Paired Top-attention/Random-real-key compensation ablation."""

import math
from typing import Dict, Tuple

import torch

from .base import CompactionAlgorithm, evaluate_compaction
from .highest_attention_keys import HighestAttentionKeysCompaction
from ..analysis import analyze_key_selections


class KeySelectionAblationCompaction(CompactionAlgorithm):
    """Select Top-RMS or Random real keys and independently control beta/value fitting."""

    def __init__(
        self,
        selection_method: str,
        use_fitted_bias: bool,
        value_method: str,
        nnls_iters: int = 2,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = "lstsq",
        c2_ridge_scale: str = "spectral",
        key_seed: int = 0,
        enable_key_selection_analysis: bool = False,
        dead_slot_threshold: float = 1e-4,
        routing_spectrum_topk: int = 20,
        hungarian_max_slots: int = 512,
    ):
        if selection_method not in {"top", "random"}:
            raise ValueError("selection_method must be 'top' or 'random'")
        if value_method not in {"direct", "fitted"}:
            raise ValueError("value_method must be 'direct' or 'fitted'")
        self.selection_method = selection_method
        self.use_fitted_bias = bool(use_fitted_bias)
        self.value_method = value_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.key_seed = int(key_seed)
        self.enable_key_selection_analysis = bool(enable_key_selection_analysis)
        self.dead_slot_threshold = dead_slot_threshold
        self.routing_spectrum_topk = routing_spectrum_topk
        self.hungarian_max_slots = hungarian_max_slots
        self._selection_context = (0, 0, 0)
        self.last_key_selection_analysis = None

    def name(self) -> str:
        bias = "fitted_beta" if self.use_fitted_bias else "zero_beta"
        return f"KeySelectionAblation_{self.selection_method}_{bias}_{self.value_method}_value"

    def set_selection_context(self, article_id: int, layer_idx: int, head_idx: int) -> None:
        self._selection_context = (int(article_id), int(layer_idx), int(head_idx))

    def _derived_seed(self) -> int:
        article_id, layer_idx, head_idx = self._selection_context
        return int((self.key_seed + article_id * 1_000_003 + layer_idx * 10_007 + head_idx) % (2**63 - 1))

    def _top_selector(self, beta_method: str = "nnls") -> HighestAttentionKeysCompaction:
        return HighestAttentionKeysCompaction(
            nnls_iters=self.nnls_iters,
            nnls_lower_bound=self.nnls_lower_bound,
            nnls_upper_bound=self.nnls_upper_bound,
            score_method="rms",
            c2_method="lsq",
            beta_method=beta_method,
            c2_ridge_lambda=self.c2_ridge_lambda,
            c2_solver=self.c2_solver,
            c2_ridge_scale=self.c2_ridge_scale,
        )

    def _select_random_and_fit_beta(
        self, K: torch.Tensor, queries: torch.Tensor, t: int, attention_bias: torch.Tensor = None,
        fit_beta: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        device = K.device
        generator = torch.Generator(device=device)
        generator.manual_seed(self._derived_seed())
        indices_tensor = torch.randperm(K.shape[0], generator=generator, device=device)[:t]
        C1 = K[indices_tensor]

        if not fit_beta:
            return C1, K.new_zeros(t), indices_tensor.cpu().tolist()

        d = K.shape[1]
        scores = (queries @ K.T).float() * (d ** -0.5)
        if attention_bias is not None:
            scores = scores + torch.broadcast_to(attention_bias.float(), scores.shape)
        maximum = scores.max(dim=1, keepdim=True).values
        exp_scores = torch.exp(scores - maximum)
        design = exp_scores[:, indices_tensor]
        target = exp_scores.sum(dim=1)
        B = self._nnls_pg(
            design, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
        )
        beta = torch.log(B).to(K.dtype)
        return C1, beta, indices_tensor.cpu().tolist()

    def _fitted_value(self, C1, beta, K, V, queries, indices, attention_bias):
        return self._compute_C2_with_method(
            C1, beta, K, V, queries,
            method="lsq", indices=indices, attention_bias=attention_bias,
            ridge_lambda=self.c2_ridge_lambda, solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale,
        )

    def _build_variants(self, C1, fitted_beta, indices, K, V, queries, attention_bias) -> Dict:
        zero_beta = torch.zeros_like(fitted_beta)
        direct_value = self._direct_C2(C1, K, V, indices)
        no_bias_fitted_value = self._fitted_value(
            C1, zero_beta, K, V, queries, indices, attention_bias
        )
        bias_fitted_value = self._fitted_value(
            C1, fitted_beta, K, V, queries, indices, attention_bias
        )
        return {
            "raw_bias_direct_value": (zero_beta, direct_value),
            "fitted_bias_direct_value": (fitted_beta, direct_value),
            "zero_bias_fitted_value": (zero_beta, no_bias_fitted_value),
            "fitted_bias_fitted_value": (fitted_beta, bias_fitted_value),
        }

    def _build_active_variant(self, C1, fitted_beta, indices, K, V, queries, attention_bias):
        beta = fitted_beta if self.use_fitted_bias else torch.zeros_like(fitted_beta)
        if self.value_method == "direct":
            value = self._direct_C2(C1, K, V, indices)
        else:
            # Critically, zero-bias fitted values are solved from zero-bias routing.
            value = self._fitted_value(C1, beta, K, V, queries, indices, attention_bias)
        return beta, value

    def compute_compacted_cache(
        self, K: torch.Tensor, V: torch.Tensor, queries: torch.Tensor, t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        cuda_devices = []
        if K.device.type == "cuda":
            cuda_devices = [K.device.index if K.device.index is not None else torch.cuda.current_device()]
        # Existing NNLS power iteration consumes torch RNG. Restore the global
        # stream afterwards so fitting choices cannot perturb answer sampling.
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            return self._compute_compacted_cache_impl(K, V, queries, t, attention_bias)

    def _compute_compacted_cache_impl(
        self, K: torch.Tensor, V: torch.Tensor, queries: torch.Tensor, t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        if t < 0 or t > K.shape[0]:
            raise ValueError(f"t must satisfy 0 <= t <= {K.shape[0]}, got {t}")
        if t == 0:
            return K[:0], K.new_empty(0), V[:0], []

        top_keys = top_beta = top_indices = None
        random_keys = random_beta = random_indices = None
        need_fitted_beta = self.use_fitted_bias or self.enable_key_selection_analysis
        if self.selection_method == "top" or self.enable_key_selection_analysis:
            top_selector = self._top_selector("nnls" if need_fitted_beta else "zero")
            top_keys, top_beta, top_indices = top_selector._select_keys_highest_attention(
                K, queries, t, attention_bias
            )
        if self.selection_method == "random" or self.enable_key_selection_analysis:
            random_keys, random_beta, random_indices = self._select_random_and_fit_beta(
                K, queries, t, attention_bias, fit_beta=need_fitted_beta
            )

        selected_keys, fitted_beta, selected_indices = (
            (top_keys, top_beta, top_indices)
            if self.selection_method == "top"
            else (random_keys, random_beta, random_indices)
        )
        beta, C2 = self._build_active_variant(
            selected_keys, fitted_beta, selected_indices, K, V, queries, attention_bias
        )

        if self.enable_key_selection_analysis:
            top_variants = self._build_variants(
                top_keys, top_beta, top_indices, K, V, queries, attention_bias
            )
            random_variants = self._build_variants(
                random_keys, random_beta, random_indices, K, V, queries, attention_bias
            )
            analysis = analyze_key_selections(
                top_keys, random_keys,
                torch.tensor(top_indices, device=K.device),
                torch.tensor(random_indices, device=K.device),
                queries, top_beta, random_beta, K.shape[0],
                dead_slot_threshold=self.dead_slot_threshold,
                spectrum_topk=self.routing_spectrum_topk,
                hungarian_max_slots=self.hungarian_max_slots,
            )
            analysis["local_indices"] = {
                "top": [int(i) for i in top_indices],
                "random": [int(i) for i in random_indices],
            }
            analysis["combination_train_metrics"] = {}
            for selection, keys, variants in (
                ("top", top_keys, top_variants), ("random", random_keys, random_variants)
            ):
                analysis["combination_train_metrics"][selection] = {}
                for name, (variant_beta, variant_value) in variants.items():
                    metrics = evaluate_compaction(
                        K, V, keys, variant_beta, variant_value, queries,
                        attention_bias=attention_bias,
                    )
                    analysis["combination_train_metrics"][selection][name] = {
                        key: (float(value) if math.isfinite(float(value)) else None)
                        for key, value in metrics.items()
                    }
            self.last_key_selection_analysis = analysis

        return selected_keys, beta, C2, selected_indices
