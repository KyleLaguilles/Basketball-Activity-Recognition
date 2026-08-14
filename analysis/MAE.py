"""
Duration MAE for Hang-Time HAR.

Total predicted time-per-class vs ground-truth time-per-class -- the metric
Brandon's doc says the downstream injury goal depends on. Prints a table and
writes two CSVs (summary + per-session).

Input format (one file per LOSO fold / per saved test set):
    predictions_best_<id>_<name>.csv  ->  pd.DataFrame(val_output).to_csv(...)
    column ","  = window index (ignored)
    column "0"  = PREDICTION   (val_output[:, 0])
    column "1"  = GROUND TRUTH (val_output[:, 1])
Orientation confirmed from src/model/validation.py: sklearn is called as
    precision_score(val_output[:, 1], val_output[:, 0])  ==  (y_true, y_pred)

Time convention: windows are 1.0 s but overlap 50%, so each window advances
only 0.5 s of NEW time. We attribute STRIDE_S to each window so the timeline
tiles without double-counting. Using the 1.0 s window length would inflate 2x.
"""

import glob
import os
import numpy as np
import pandas as pd

# ---- params (from your training command) ----
SW_LENGTH = 1.0
OVERLAP = 0.50
STRIDE_S = SW_LENGTH * (1 - OVERLAP)          # 0.5 s of new time per window
NUM_CLASSES = 9
CLASSES = ['dribbling', 'shot', 'pass', 'rebound', 'layup',
           'walking', 'running', 'standing', 'sitting']

# ---- orientation (verified against validation.py: col 1 = truth, col 0 = pred) ----
TRUE_COL, PRED_COL = '1', '0'

# ---- if the split saver is later changed to write a per-window session id ----
# ---- column, set its name here and a single pooled file will be split by it ----
SESSION_COL = None

# ---- one run dir per seed ----
# loso  : point at the run folder (14 per-subject files -> per-subject breakdown)
# split : point at the run folder for the global game number, OR at its
#         'session_preds/' subfolder (after the validation.py patch) for per-subject
RUN_DIRS = [
    'logs/session_specific/split_DandWvsG/inceptioncontext/2026-04-02_19-32-23',
    'logs/session_specific/split_DandWvsG/inceptioncontext/2026-04-02_19-45-54',
    'logs/session_specific/split_DandWvsG/inceptioncontext/2026-04-02_19-56-11',
]
FILE_GLOB = 'predictions_best_*.csv'
OUTPUT_DIR = '.'


def session_id(path, multi):
    base = os.path.basename(path).replace('predictions_best_', '')
    return base.split('_')[0] if multi else 'all_games'


def load_file(path):
    df = pd.read_csv(path, index_col=0)
    yt = df[TRUE_COL].to_numpy(int)
    yp = df[PRED_COL].to_numpy(int)
    sess = df[SESSION_COL].to_numpy() if SESSION_COL and SESSION_COL in df.columns else None
    return yt, yp, sess


def seconds(y):
    return np.bincount(y, minlength=NUM_CLASSES) * STRIDE_S


def sessions_for_run(run_dir):
    """List of (session_id, t_true[9], t_pred[9]) for one seed/run."""
    files = sorted(glob.glob(os.path.join(run_dir, FILE_GLOB)))
    multi = len(files) > 1
    out = []
    for f in files:
        yt, yp, sess = load_file(f)
        if sess is not None:                       # split a pooled file by session column
            for s in pd.unique(sess):
                m = sess == s
                out.append((str(s), seconds(yt[m]), seconds(yp[m])))
        else:                                       # one file == one session / fold
            out.append((session_id(f, multi), seconds(yt), seconds(yp)))
    return out


def recall_check(run_dir):
    """Pooled per-class recall, to confirm column orientation vs cp_scores_rec."""
    tp = np.zeros(NUM_CLASSES)
    sup = np.zeros(NUM_CLASSES)
    for f in sorted(glob.glob(os.path.join(run_dir, FILE_GLOB))):
        yt, yp, _ = load_file(f)
        for c in range(NUM_CLASSES):
            m = yt == c
            sup[c] += m.sum()
            tp[c] += (yp[m] == c).sum()
    return np.divide(tp, sup, out=np.full(NUM_CLASSES, np.nan), where=sup > 0)


def main():
    per_seed = [s for s in (sessions_for_run(d) for d in RUN_DIRS) if s]
    if not per_seed:
        print('no prediction files found -- check RUN_DIRS / FILE_GLOB')
        return

    # summary: per-session error averaged over sessions, then over seeds
    mae_s, bias_s, true_s = [], [], []
    for sessions in per_seed:
        err = np.array([np.abs(tp - tt) for _, tt, tp in sessions])
        sgn = np.array([tp - tt for _, tt, tp in sessions])
        tru = np.array([tt for _, tt, tp in sessions])
        mae_s.append(err.mean(0))
        bias_s.append(sgn.mean(0))
        true_s.append(tru.mean(0))
    mae = np.mean(mae_s, 0)
    bias = np.mean(bias_s, 0)
    true_tot = np.mean(true_s, 0)

    # per-session table, averaged across seeds sharing a session id
    bucket = {}
    for sessions in per_seed:
        for sid, tt, tp in sessions:
            bucket.setdefault(sid, []).append((tt, tp))
    rows = []
    for sid, lst in bucket.items():
        tt = np.mean([a for a, _ in lst], 0)
        tp = np.mean([b for _, b in lst], 0)
        for ci, c in enumerate(CLASSES):
            rows.append(dict(session=sid, cls=c, true_s=round(tt[ci], 2),
                             pred_s=round(tp[ci], 2),
                             abs_err_s=round(abs(tp[ci] - tt[ci]), 2),
                             bias_s=round(tp[ci] - tt[ci], 2)))
    by_session = pd.DataFrame(rows)
    summary = pd.DataFrame(dict(
        cls=CLASSES, mae_s=mae.round(2), bias_s=bias.round(2),
        true_s=true_tot.round(2),
        pct_of_true=np.where(true_tot > 0, (100 * mae / true_tot).round(1), np.nan)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sp = os.path.join(OUTPUT_DIR, 'duration_mae_summary.csv')
    bp = os.path.join(OUTPUT_DIR, 'duration_mae_by_session.csv')
    summary.to_csv(sp, index=False)
    by_session.to_csv(bp, index=False)

    rec = recall_check(RUN_DIRS[0])
    print('recall sanity (compare to cp_scores_rec_*.csv): '
          + '  '.join(f'{c[:4]}={r:.2f}' for c, r in zip(CLASSES, rec)))
    print(f'\nmacro duration MAE: {mae.mean():.1f} s   '
          f'({len(per_seed)} seed(s), {len(bucket)} session(s))')
    print('%-10s %9s %9s %9s' % ('class', 'MAE(s)', 'bias(s)', '%of_true'))
    print('-' * 41)
    for c, m, b, t in zip(CLASSES, mae, bias, true_tot):
        print('%-10s %9.1f %+9.1f %8s'
              % (c, m, b, f'{100*m/t:.0f}%' if t > 0 else 'n/a'))
    print(f'\nsaved:\n  {sp}\n  {bp}')


if __name__ == '__main__':
    main()