# Scoped prior-art check: KV-cache compression fitting objectives (2024–2026)

The bundled multi-source search command was run for `KV cache compression distillation objective attention output hidden state logits reconstruction` over 2024–2026. It did not return within approximately 70 seconds and was terminated without per-source output. The source-specific tables below therefore record the unavailable CLI result; a primary-source fallback search is reported afterward.

### Semantic Scholar (unavailable)

No results were returned before the multi-source command timed out.

### OpenAlex (unavailable)

No results were returned before the multi-source command timed out.

### arXiv (unavailable from CLI)

No CLI results were returned before timeout. Verified fallback results appear below.

### OpenReview (unavailable from CLI)

No CLI results were returned before timeout. Verified fallback results appear below.

### Crossref (unavailable)

No results were returned before the multi-source command timed out.

### DBLP (unavailable)

No results were returned before the multi-source command timed out.

### Primary-source fallback (5 papers)

| # | Title | Date | Venue/status | Citations |
|---|---|---|---|---|
| [1](https://arxiv.org/abs/2503.10337) | KV-Distill: Nearly Lossless Learnable Context Compression for LLMs | 2025-03 | arXiv | unavailable |
| [2](https://arxiv.org/abs/2602.16284) | Fast KV Compaction via Attention Matching | 2026-02, v2 2026-05 | arXiv | unavailable |
| [3](https://arxiv.org/abs/2603.27819) | KVSculpt: KV Cache Compression as Distillation | 2026-03 | arXiv | unavailable |
| [4](https://openreview.net/forum?id=klmc4fwPLd) | Value-Guided KV Compression for LLMs via Approximated CUR Decomposition | 2025-09, modified 2026-04 | NeurIPS 2025 poster | unavailable |
| [5](https://openreview.net/forum?id=Ql0G1Zsobn) | Retrospective Sparse Attention for Efficient Long-Context Generation | 2026 | OpenReview | unavailable |

## Summary of searched results

### 1. Overview

The scoped check covers fitting targets for sequence-length KV-cache compression from 2024 through 2026. Five directly relevant papers were verified through primary arXiv/OpenReview pages after the bundled aggregator timed out.

### 2. Trends

The literature is moving from attention-score eviction toward functional reconstruction. The target hierarchy now spans attention mass/output matching, value-aware attention-output reconstruction, and end-to-end next-token KL distillation. Recent work also studies continuous latent KV optimization and long-decode error accumulation.

### 3. Key themes

1. Local attention-function preservation: Attention Matching and KVSculpt preserve per-layer attention behavior ([2], [3]).
2. End-to-end predictive distillation: KV-Distill matches compressed and full-cache token distributions with a KL-type loss ([1]).
3. Value-aware selection: CurDKV argues that attention weights alone do not guarantee attention-output preservation ([4]).
4. Temporal error correction: retrospective methods address cumulative errors during long decoding ([5]).

### 4. Keywords frequency

| Keyword | Count |
|---|---:|
| KV cache compression/compaction | 5 |
| Attention reconstruction/behavior | 3 |
| Distillation | 2 |
| Value-aware | 2 |
| Long-context decoding | 2 |

### 5. Most cited by accepted paper

Reliable citation counts were unavailable from the completed fallback sources, so no ranking is reported.

### 6. Most cited by first author

Reliable citation counts were unavailable from the completed fallback sources, so no ranking is reported.

### 7. Recommendations for reading

1. [Attention Matching](https://arxiv.org/abs/2602.16284): establishes the mass-plus-local-output objective and its closed-form decomposition.
2. [KV-Distill](https://arxiv.org/abs/2503.10337): establishes next-token KL as an end-to-end alternative, but with training/optimization cost.
3. [Value-Guided KV Compression](https://openreview.net/forum?id=klmc4fwPLd): motivates incorporating value contributions into selection rather than fitting mass alone.
4. [KVSculpt](https://arxiv.org/abs/2603.27819): tests unconstrained continuous keys with least-squares values and adaptive budgets.
