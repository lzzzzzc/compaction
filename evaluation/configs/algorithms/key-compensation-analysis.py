"""Top-RMS vs Random-real-key bias/value compensation factorial."""

import math


COMMON = {
    'algorithm': 'key_selection_ablation',
    'nnls_iters': 2,
    'nnls_lower_bound': math.exp(-3),
    'nnls_upper_bound': math.exp(3),
    'c2_ridge_lambda': 0,
    'c2_solver': 'lstsq',
    'c2_ridge_scale': 'spectral',
    'key_seed': 0,
    'dead_slot_threshold': 1e-4,
    'routing_spectrum_topk': 20,
    'hungarian_max_slots': 512,
}


def variant(selection, fitted_bias, value_method, analysis=False):
    return {
        **COMMON,
        'selection_method': selection,
        'use_fitted_bias': fitted_bias,
        'value_method': value_method,
        'enable_key_selection_analysis': analysis,
    }


config = {
    'key_analysis_top_raw_direct': variant('top', False, 'direct'),
    'key_analysis_random_raw_direct': variant('random', False, 'direct'),
    'key_analysis_top_beta_direct': variant('top', True, 'direct'),
    'key_analysis_random_beta_direct': variant('random', True, 'direct'),
    'key_analysis_top_zero_fitted': variant('top', False, 'fitted'),
    'key_analysis_random_zero_fitted': variant('random', False, 'fitted'),
    # The complete Top configuration owns the paired mechanism diagnostics.
    'key_analysis_top_beta_fitted': variant('top', True, 'fitted', analysis=True),
    'key_analysis_random_beta_fitted': variant('random', True, 'fitted'),
}
