# Pre-Registration: Risk-Proxy Sensitivity Analysis

**Date:** 2026-09-09
**Author:** Kyle Laguilles
**Status:** Draft — pending advisor review

---

## 1. Purpose

This analysis demonstrates how classifier bias in per-session activity durations propagates into a downstream injury-risk assessment. The risk proxy defined here is **illustrative, not validated**: it is a sensitivity vehicle showing that the choice of HAR method (windowed vs. dense) changes risk stratification under any plausible weighting of basketball activities.

This analysis does **not** predict injuries. No injury labels exist in the dataset, and BOracle (the downstream injury-risk platform) does not yet exist as code. The claim is limited to: classifier duration bias changes the output of a monotone risk function, and the magnitude of that change depends on which classes carry the most risk weight.

---

## 2. Risk Proxy Definition

### 2.1 Tier Assignments

Each of HTH's nine activity classes is assigned to one of three ordinal tiers based on published injury epidemiology and biomechanical load research:

| Tier | Classes | Justification | Citations |
|------|---------|---------------|-----------|
| 3 (landing/explosive) | rebound, layup, shot | Rebounding is the most common activity at time of ankle injury in NCAA basketball (men: 34.4%, women: 30.3%). Shooting accounts for 10.6–11.6% of injuries. Landing mechanics during aerial catching produce stiffer joints and larger knee abduction moments (ACL risk). | Tummala et al. 2018 (OJSM); Lytle et al. 2019; J Appl Biomech 2025 |
| 2 (locomotion/load) | running, dribbling, pass | Sprints and high-intensity specific movements produce ~0.44 AU PlayerLoad per event (~20 AU/min intensity), reflecting meaningful cumulative mechanical stress. Ball handling/dribbling appears at the bottom of Tummala's activity-at-injury distribution. | Pernigoni et al. 2021 (Frontiers Psychol); Tummala et al. 2018 Fig 3 |
| 1 (rest/low) | walking, standing, sitting | These activities do not appear in Tummala's activity-at-injury categories. They produce negligible mechanical load and no landing impact. | Absence from injury data; definitional |

**Note on layup:** Layup is not a separate category in Tummala et al. (likely absorbed into "shooting" or "general play"). It is assigned to Tier 3 based on shared landing mechanics with rebounding rather than direct epidemiological evidence. This is stated explicitly in the paper.

**Note on Pernigoni and tier ordering:** Pernigoni shows that jumps produce *lower* per-event PlayerLoad (0.22 AU) than sprints/HSMs (0.44 AU). This does not contradict the Tier 3 > Tier 2 ordering because PlayerLoad measures whole-body inertial acceleration (volume), not the joint-specific impact forces that drive ankle/knee injuries on landing. The Tier 3 > Tier 2 ordering is grounded in injury epidemiology (Tummala), not in PlayerLoad.

**Note on "general play":** Tummala's second-largest category is "general play" (men: 23.6%, women: 28.6%), a catch-all that may contain fragments of running, passing, defending, and walking. We do not assign it to a tier. Our tiering uses only the specific named activity categories that map to HTH classes.

### 2.2 Risk Score Formula

For each session *s*, the risk score under condition *c* ∈ {truth, windowed, dense} is:

```
R_c(s) = Σ_{k=1}^{9} w_{tier(k)} × f_{c,k}(s)
```

where:
- *k* indexes the nine activity classes
- *tier(k)* ∈ {1, 2, 3} is the tier assignment for class *k*
- *w_t* is the weight for tier *t*
- *f_{c,k}(s)* is the fraction of session *s* time spent in class *k* under condition *c*

Fractions are used rather than raw durations to normalize for session length.

### 2.3 Weight Vectors

Five weight vectors are tested, all preserving the strict tier ordering w₃ > w₂ > w₁ ≥ 0:

| Label | w₃ (landing) | w₂ (locomotion) | w₁ (rest) | Rationale |
|-------|-------------|----------------|----------|-----------|
| A: linear | 3 | 2 | 1 | Minimal separation between tiers |
| B: moderate | 5 | 2 | 1 | Moderate landing emphasis |
| C: strong | 10 | 3 | 1 | Strong landing emphasis |
| D: compressed | 5 | 3 | 1 | Narrow gap between Tier 3 and Tier 2 |
| E: zero-rest | 3 | 2 | 0 | Rest contributes nothing to risk |

The analysis claims robustness if and only if the primary finding holds under **all five** vectors.

---

## 3. Conditions

Three conditions are compared:

1. **Ground truth:** Per-session per-class duration fractions computed from manual annotations.
2. **Windowed baseline:** Per-session per-class duration fractions computed from windowed (1-second, majority-vote) predictions.
3. **Dense prediction:** Per-session per-class duration fractions computed from dense (per-sample) predictions.

An optional **calibrated** variant applies the ship-ready LOO calibration correction factors (pass, walking, standing only) to windowed and dense predictions before computing scores. Calibration is applied as a secondary analysis, not the primary comparison.

---

## 4. Sample Sizes

- **Paired panel (n=10):** 5 subjects × 2 capture days. Windowed, dense, and truth durations are available for the same sessions. This is the primary comparison surface.
- **Dense-only panel (n=24):** All 14 subjects across both capture events. Dense and truth durations available; windowed not available for all sessions. Used for dense-vs-truth rank agreement only.

These match the sample sizes used in the rank-stability and duration-bias analyses.

---

## 5. Pre-Registered Metrics

### 5.1 Primary Metrics (n=10 paired)

**(a) Rank agreement with truth (Spearman ρ):**
For each weight vector, compute the Spearman rank correlation between truth risk scores and windowed risk scores, and between truth risk scores and dense risk scores. Report ρ and p-value for each.

**(b) Tertile flips:**
Assign each session to a risk tertile (low / medium / high) based on truth risk scores. Count how many sessions change tertile under windowed predictions vs. under dense predictions. At n=10, tertiles are sized 3/4/3 or 4/3/3.

**(c) Mean absolute score deviation (MASD):**
```
MASD_c = (1/n) Σ_s |R_c(s) - R_truth(s)|
```
Reported for windowed and dense, under each weight vector.

**(d) Mean relative score deviation (MRSD):**
```
MRSD_c = (1/n) Σ_s |R_c(s) - R_truth(s)| / R_truth(s)
```
Guards against absolute deviation being dominated by high-scoring sessions.

**(e) Class-level decomposition:**
For the weight vector with the largest MASD difference between windowed and dense, decompose the score error by class: which class-level duration errors contribute most to the risk-score disagreement? Report as a 9-class bar chart of signed weighted error contribution.

### 5.2 Secondary Metrics

**(f) Calibration effect (n=10 paired):**
Repeat metrics (a)–(d) with ship-ready calibration corrections applied to pass, walking, and standing. Report the change in MASD and MRSD. Does calibration reduce risk-score error? Does it change any tertile assignments?

**(g) Dense-only rank agreement (n=24):**
Spearman ρ between truth and dense risk scores across all 24 sessions, under each weight vector. This tests whether the dense-vs-truth agreement holds at the larger sample size where rank stability previously succeeded for rebound (ρ=0.544, p=0.006).

---

## 6. Success Bars

### Primary success criterion:
Under **all five** weight vectors, dense prediction produces risk scores **closer to ground truth** than windowed prediction, as measured by:
- Higher Spearman ρ with truth (or both high), AND
- Fewer tertile flips from truth (or both zero), AND
- Lower MASD

If all three hold across all five vectors: **robust claim.** The paper states: "Under any weighting where landing actions outweigh locomotion outweigh rest, dense prediction preserves risk stratification better than the windowed baseline."

### Partial success:
If the claim holds under 3–4 of 5 vectors but fails under 1–2: **conditional claim.** Report the weight-vector dependence. Examine which tier gap drives the failure (likely the narrow-gap vector D).

### Null result:
If windowed and dense produce similar risk scores across most vectors: the duration errors, while real, do not propagate meaningfully into risk assessment under this proxy. The paper reports this honestly — the measurement-validity story (duration bias, rank stability) stands, but the "so what" for injury risk is weaker. The risk proxy section becomes a brief negative result rather than a main finding.

---

## 7. Failure Interpretations

| Outcome | Interpretation | Paper consequence |
|---------|---------------|-------------------|
| Dense wins on all 5 vectors | Duration bias propagates into risk stratification; dense fixes it | Main finding; 1–2 paragraphs + robustness table |
| Dense wins on 3–4 vectors | Propagation depends on relative tier weighting | Conditional finding; state which weightings fail |
| No difference (MASD similar) | Duration errors too small or cancel across classes | Brief negative result; paper scope stays at measurement validity |
| Windowed wins on some vectors | Unexpected; investigate whether windowed's overcount of pass/walking accidentally improves Tier 1/2 scores | Investigate before reporting; may indicate proxy design issue |
| Calibration eliminates the gap | Ship-ready corrections are sufficient; the risk-propagation story is about correctable bias | Positive result for calibration; state that simple corrections suffice for risk assessment |

---

## 8. Script Specification

**File:** `analysis/risk_proxy_sensitivity.py`

**Dependencies:** Reuses `_session_common.py` (or the equivalent shared session-loading utilities used by `boracle_duration_bias.py` and the rank-stability script).

**Inputs:**
- Ground-truth per-session per-class duration files
- Windowed prediction per-session per-class duration files (n=10)
- Dense prediction per-session per-class duration files (n=10 paired + n=24 full)
- Calibration correction factors for pass, walking, standing (from LOO analysis)

**Configuration (top of script):**
```python
TIER_MAP = {
    "rebound": 3, "layup": 3, "shot": 3,
    "running": 2, "dribbling": 2, "pass": 2,
    "walking": 1, "standing": 1, "sitting": 1,
}

WEIGHT_VECTORS = {
    "A_linear":     {3: 3,  2: 2, 1: 1},
    "B_moderate":   {3: 5,  2: 2, 1: 1},
    "C_strong":     {3: 10, 2: 3, 1: 1},
    "D_compressed": {3: 5,  2: 3, 1: 1},
    "E_zero_rest":  {3: 3,  2: 2, 1: 0},
}
```

**Outputs:**
- Console table: per-weight-vector Spearman ρ (windowed vs truth, dense vs truth), MASD, MRSD, tertile flip counts
- Console table: calibrated variant of the above
- Console table: n=24 dense-only Spearman ρ per weight vector
- Class decomposition bar data for the highest-contrast weight vector
- Hard-fail assertion: all weight vectors satisfy w₃ > w₂ > w₁ ≥ 0

**Flags:**
- `--calibrate`: apply ship-ready correction factors before scoring
- `--dense-only`: run n=24 panel (metrics f, g only)

---

## 9. Amendments

(None yet. Amendments will be documented here if environment or data issues require changes after pre-registration, following the same protocol as `prereg_dense.md`.)