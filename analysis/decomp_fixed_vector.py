"""
Class decomposition of the exposure-index error under ONE fixed weight vector, plus the
explosive-action (tier 3) time sums quoted in the paper's Results G.

Why this exists: risk_proxy_sensitivity.py prints its [6] decomposition only for the
vector with the largest windowed/dense MASD gap, which was C_strong (10,3,1) on seed 1
but E_zero_rest on seeds 2 and 3. The paper quotes the (10,3,1) decomposition for all
three seeds, so it has to be computed with the vector held fixed. Nothing else changes:
the session loading, pairing, fractions and weights are imported from
risk_proxy_sensitivity.py and _session_common.py, not reimplemented.

Read-only, CPU-only, run from the repo root.

Usage (one seed per call):
    python analysis/decomp_fixed_vector.py \
        --windowed_dir logs/subset_specific/loso_G/inceptioncontext/2026-09-11_18-13-22_24fold_windowed_seed1 \
        --dense_dir    logs/subset_specific/loso_G/inceptioncontext/2026-09-11_12-06-12_24fold_dense_seed1 \
        --anchor_masd_win 0.1203 --anchor_masd_den 0.0835

The anchors are the C_strong MASD values that risk_proxy_sensitivity.py printed for the
same run pair (its section [4]). If the recomputed MASD does not reproduce them, the
sessions were loaded differently and nothing below is comparable, so the script stops.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _session_common as sc                 # noqa: E402
import risk_proxy_sensitivity as rps         # noqa: E402  -- main() is __main__-guarded

CLASS_NAMES = rps.CLASS_NAMES
TIER_MAP = rps.TIER_MAP
fail = rps.fail
VECTOR_LABEL = "C_strong"                    # (w3, w2, w1) = (10, 3, 1)
ANCHOR_TOL = 5e-4
RULE = "=" * 100


def compute(paired, w):
    """Everything this script reports, from paired sessions and a class-weight vector."""
    recs = sorted(paired)
    truth_f = np.array([rps.fractions(paired[r]["true"]) for r in recs])
    out = {"recs": recs, "truth_f": truth_f, "contrib": {}, "masd": {}}
    for key in ("win_est", "den_est"):
        est_f = np.array([rps.fractions(paired[r][key]) for r in recs])
        out["contrib"][key] = w * (est_f - truth_f).mean(axis=0)
        out["masd"][key] = rps.masd(est_f @ w, truth_f @ w)

    explosive = [CLASS_NAMES.index(c) for c in sc.EXPLOSIVE]
    t3 = {k: np.array([sc.secs(np.asarray(paired[r][k])[explosive].sum()) for r in recs])
          for k in ("true", "win_est", "den_est")}
    out["t3"] = t3
    out["t3_stats"] = {}
    for key in ("win_est", "den_est"):
        pooled = (t3[key].sum() - t3["true"].sum()) / t3["true"].sum() * 100.0
        mae = float(np.mean(np.abs(t3[key] - t3["true"])))
        rho, p = rps.safe_spearman(t3["true"], t3[key])
        out["t3_stats"][key] = (pooled, mae, rho, p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windowed_dir", required=True)
    ap.add_argument("--dense_dir", required=True)
    ap.add_argument("--windowed_npz_pattern", default=None)
    ap.add_argument("--dense_npz_pattern", default=None)
    ap.add_argument("--anchor_masd_win", type=float, default=None,
                    help="C_strong windowed MASD printed by risk_proxy_sensitivity.py [4].")
    ap.add_argument("--anchor_masd_den", type=float, default=None,
                    help="C_strong dense MASD printed by risk_proxy_sensitivity.py [4].")
    sc.add_session_args(ap)
    args = ap.parse_args()

    rps.check_tier_map()
    rps.check_weight_vectors()
    w = rps.class_weights(rps.WEIGHT_VECTORS[VECTOR_LABEL])

    labels_df, segments = sc.load_common(args)
    win_arrays, _ = sc.load_path(args.windowed_dir, labels_df, segments, args.labels,
                                 args.seam_map, dense=False,
                                 npz_pattern=args.windowed_npz_pattern,
                                 allow_partial=args.allow_partial_folds)
    den_arrays, _ = sc.load_path(args.dense_dir, labels_df, segments, args.labels,
                                 args.seam_map, dense=True,
                                 npz_pattern=args.dense_npz_pattern,
                                 allow_partial=args.allow_partial_folds)
    paired = sc.paired_sessions(win_arrays, den_arrays, segments)
    n = sc.check_session_count(paired, args.allow_partial_folds, "paired: ")

    r = compute(paired, w)

    print(RULE)
    print(f"FIXED-VECTOR DECOMPOSITION -- {VECTOR_LABEL} (w3, w2, w1) = "
          f"({rps.WEIGHT_VECTORS[VECTOR_LABEL][3]}, {rps.WEIGHT_VECTORS[VECTOR_LABEL][2]}, "
          f"{rps.WEIGHT_VECTORS[VECTOR_LABEL][1]}), n = {n} paired sessions")
    print(RULE)
    print(f"  windowed: {args.windowed_dir}")
    print(f"  dense   : {args.dense_dir}\n")

    print("[1] ANCHOR -- MASD under this vector, vs risk_proxy_sensitivity.py [4]")
    for key, anchor, label in (("win_est", args.anchor_masd_win, "windowed"),
                               ("den_est", args.anchor_masd_den, "dense")):
        got = r["masd"][key]
        if anchor is None:
            print(f"  {label:<9} MASD {got:.4f}   (no anchor given -- not certified)")
            continue
        verdict = "MATCH" if abs(got - anchor) <= ANCHOR_TOL else "MISMATCH"
        print(f"  {label:<9} MASD {got:.4f}   anchor {anchor:.4f}   {verdict}")
        if verdict == "MISMATCH":
            fail(f"{label} MASD {got:.4f} does not reproduce the anchor {anchor:.4f}; the "
                 "sessions were loaded differently from risk_proxy_sensitivity.py.")

    print("\n[2] CLASS DECOMPOSITION -- mean signed contribution to the score deviation")
    print(f"  {'class':<11} {'tier':>4} {'w_k':>5} {'true_frac':>10} {'windowed':>11} {'dense':>11}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<11} {TIER_MAP[name]:>4} {w[c]:>5g} {r['truth_f'][:, c].mean():>10.4f} "
              f"{r['contrib']['win_est'][c]:>+11.4f} {r['contrib']['den_est'][c]:>+11.4f}")
    cw, cd = r["contrib"]["win_est"], r["contrib"]["den_est"]
    print(f"  {'signed total':<33} {cw.sum():>+11.4f} {cd.sum():>+11.4f}")
    print(f"  {'sum of |contributions|':<33} {np.abs(cw).sum():>11.4f} {np.abs(cd).sum():>11.4f}")
    print(f"  ratio of |contribution| sums, windowed / dense: "
          f"{np.abs(cw).sum() / np.abs(cd).sum():.2f}")

    print("\n[3] EXPLOSIVE-ACTION TIME (shot + rebound + layup), per session, in seconds")
    print(f"  true total over sessions: {r['t3']['true'].sum():.1f} s")
    print(f"  {'':<9} {'pooled bias':>12} {'mean |error| per session':>26} {'Spearman rho':>14}")
    for key, label in (("win_est", "windowed"), ("den_est", "dense")):
        pooled, mae, rho, p = r["t3_stats"][key]
        print(f"  {label:<9} {pooled:>+11.1f}% {mae:>24.2f} s {rho:>14.3f}")

    print("\nRead-only; nothing was written.")


if __name__ == "__main__":
    try:
        main()
    except sc.HardFail as exc:
        print(f"\nHARD FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
