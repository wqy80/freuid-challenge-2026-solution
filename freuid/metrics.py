from __future__ import annotations

import numpy as np


def _rates(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    order = np.argsort(scores)
    y = labels[order]
    n_attack = max(int((y == 1).sum()), 1)
    n_bona = max(int((y == 0).sum()), 1)

    thresholds = np.r_[-np.inf, np.unique(scores[order]), np.inf]
    apcer = []
    bpcer = []
    for t in thresholds:
        # Predict attack when score >= threshold.
        apcer.append(((scores[labels == 1] < t).sum()) / n_attack)
        bpcer.append(((scores[labels == 0] >= t).sum()) / n_bona)
    return np.asarray(bpcer, dtype=np.float64), np.asarray(apcer, dtype=np.float64)


def audet(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    bpcer, apcer = _rates(labels, scores)
    order = np.argsort(bpcer)
    x = bpcer[order]
    y = apcer[order]
    # Use the best APCER observed at each BPCER to avoid threshold ties inflating area.
    ux = np.unique(x)
    uy = np.asarray([y[x == v].min() for v in ux])
    return float(np.trapezoid(uy, ux))


def apcer_at_bpcer(labels, scores, target_bpcer: float = 0.01) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    bpcer, apcer = _rates(labels, scores)
    ok = bpcer <= target_bpcer + 1e-12
    if ok.any():
        return float(apcer[ok].min())
    idx = int(np.argmin(np.abs(bpcer - target_bpcer)))
    return float(apcer[idx])


def freuid_score(labels, scores) -> dict[str, float]:
    a = audet(labels, scores)
    p = apcer_at_bpcer(labels, scores, 0.01)
    g_audet = 1.0 - a
    g_apcer = 1.0 - p
    if g_audet + g_apcer <= 0:
        score = 1.0
    else:
        score = 1.0 - 2.0 * g_audet * g_apcer / (g_audet + g_apcer)
    return {
        "freuid": float(score),
        "audet": float(a),
        "apcer_at_1bpcer": float(p),
    }
