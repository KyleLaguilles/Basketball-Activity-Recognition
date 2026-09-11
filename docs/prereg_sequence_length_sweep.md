# Pre-Registration: Dense Sequence-Length Sweep

**Date:** 2026-09-09
**Author:** Kyle Laguilles
**Status:** Draft — commit before launching any sweep runs

---

## 1. Purpose

The dense prediction improvement (rebound F1 0.109 → 0.247) was obtained at `--dense_seq_len 500` (500 samples = 10 seconds at 50 Hz). The diagnostic claim is that rebound's discriminative signal lives in the run-in transition excluded by 1-second windows. This sweep tests how much temporal context the dense model actually needs by varying the sequence length while holding all other hyperparameters fixed.

The result is a small ablation table for the paper (§4 in the outline) answering: "Does rebound F1 improve with more context, and is 500 samples sufficient or a lucky middle ground?"

---

## 2. Sweep Design

Three sequence lengths, all at seed 1, all else fixed:

| Label | --dense_seq_len | Duration (s) | Context vs. baseline | Status |
|-------|----------------|-------------|---------------------|--------|
| short | 250 | 5.0 | 0.5× | NEW — requires GPU run |
| baseline | 500 | 10.0 | 1.0× | EXISTS — seed 1 result already in dense_results.md |
| long | 1000 | 20.0 | 2.0× | NEW — requires GPU run |

**Two new GPU runs required.** The baseline (500) result is reused from the existing seed-1 dense run (`2026-09-04_16-04-41_dense_b10_seed1`).

---

## 3. Fixed Conditions

All of the following are held constant across the three sweep points. Any deviation requires an amendment.

- Architecture: InceptionContext + BiLSTM (Stage 1 + Stage 2)
- Batch size: 10 (the critical fix from the dense pre-reg)
- Seed: 1
- Class weights: sqrt_inverse on pre-augmentation counts
- Augmentation: none (matching the seed-1 baseline)
- Evaluation: LOSO 5-fold (same 5 subjects as seed-1 baseline)
- Data: seam-map-corrected, same `data/seam_map.json`
- Dense flag: `--dense` enabled
- All other training hyperparameters: unchanged from seed-1 baseline cfg.txt

---

## 4. Metrics

Evaluated using the same scripts and protocols as the seed-1 baseline:

**(a) Per-class sample-level F1** via `analysis/sample_level_f1.py --dense`, with focus on:
- Rebound F1 (primary)
- Macro F1 (secondary)
- Walking and running F1 (context — long-duration classes that may respond differently to sequence length)

**(b) Rebound duration bias** via `analysis/boracle_duration_bias.py --dense`:
- Pooled duration ratio (est/true)
- Per-fold direction stability (how many folds agree on UNDER vs. OVER)

**(c) Rebound precision and recall** from sample_level_f1.py output, to distinguish whether any F1 change is driven by recall (finding more events) or precision (fewer false positives).

---

## 5. Predictions

These are committed before any sweep run is launched.

**P1 (primary — rebound F1 increases with sequence length):**
Rebound F1 at seq_len=250 < rebound F1 at seq_len=500. The run-in transition that carries rebound's discriminative signal spans several seconds; 5 seconds of context may truncate it. Direction: shorter context → worse rebound.

**P2 (secondary — diminishing returns at 1000):**
Rebound F1 at seq_len=1000 ≥ rebound F1 at seq_len=500, but the gain (if any) is smaller than the 500→250 drop. The model already captures most of the transition at 10 seconds; doubling context adds marginal signal.

**P3 (macro F1 stability):**
Macro F1 varies by less than 0.03 across the three sequence lengths. Macro F1 is dominated by high-support classes (walking, running, dribbling) whose signals are local, not context-dependent. Large macro F1 swings would indicate a training dynamics issue, not a context effect.

**P4 (long-duration class invariance):**
Walking and running F1 each vary by less than 0.02 across the three lengths. These classes have long contiguous segments and don't depend on transition context.

---

## 6. Success Bars

### 6.1 Primary bar (rebound context dependence):
Rebound F1 at seq_len=250 is **at least 0.03 lower** than at seq_len=500 (baseline: 0.2346). This confirms that context beyond 5 seconds is load-bearing for rebound classification and supports the transition-signal hypothesis.

### 6.2 Interpretive bar (sufficient context):
Rebound F1 at seq_len=1000 is **within 0.03** of seq_len=500. This confirms that 10 seconds of context is sufficient — the model has captured the transition and more context doesn't help.

### Combined interpretation:
If both bars clear: "The dense model requires approximately 10 seconds of temporal context to capture rebound's run-in transition. Halving the context degrades rebound F1; doubling it does not meaningfully improve it."

---

## 7. Failure Interpretations

| Outcome | Interpretation |
|---------|---------------|
| P1 holds, P2 holds | Context dependence confirmed, 10s is sufficient. One clean paragraph in §4. |
| P1 holds, P2 fails (1000 >> 500) | Rebound benefits from even more context than 10s. Interesting — may indicate the transition is longer than assumed, or the BiLSTM needs more runway. Report and note as future work. |
| P1 fails (250 ≈ 500) | Rebound doesn't need the extra context, or 5 seconds is already enough. Weakens the transition-signal narrative — the improvement may be about label granularity alone, not context. Re-examine whether the BiLSTM is doing the context work rather than the sequence length. |
| P1 reversed (250 > 500) | Unexpected. Longer sequences may be introducing noise or gradient issues. Investigate training curves before reporting. |
| P3 fails (macro F1 swings > 0.03) | Training dynamics issue — sequence length is affecting optimization, not just signal availability. Check batch structure and gradient updates. |
| P4 fails (walking/running shift > 0.02) | Sequence length affects classes it shouldn't. Same investigation as P3. |

---

## 8. Run Specification

**Launch commands:** Use the same launch script pattern as the seed-1 run, overriding only `--dense_seq_len`. Verify against seed-1 `cfg.txt` that no other flag differs.

**Verification before analysis:**
- `cfg.txt` diff against seed-1: only `dense_seq_len` should differ
- Fold count: 5
- Total sample count: 614,733 (must match seed-1 exactly — same data, same folds)
- Coverage: 1.00000

**Analysis commands:**
```
python analysis/sample_level_f1.py --dense --results_dir <run_dir> --labels <labels> --seam_map data/seam_map.json
python analysis/boracle_duration_bias.py --dense --results_dir <run_dir> --labels <labels> --seam_map data/seam_map.json
```

---

## 9. Paper Integration

The sweep result appears in §4 (Diagnosing the Rebound Ceiling) as a small table:

| seq_len | Rebound F1 | Macro F1 | Rebound duration ratio |
|---------|-----------|---------|----------------------|
| 250 | — | — | — |
| 500 | 0.2346 | 0.5284 | 0.7173 |
| 1000 | — | — | — |

One paragraph of interpretation following the table, connecting the result to the transition-signal hypothesis.

---

## 10. Amendments

(None yet.)