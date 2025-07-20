from __future__ import annotations

import math
from copy import deepcopy
import numpy as np

def _to_logit(p, eps = 1e-12):
    p = min(max(p, eps), 1.0-eps)
    return math.log(p) - math.log1p(-p)

def _delta(p_orig, p_alt, /, *, use_logits):
    if use_logits:
        return _to_logit(p_orig) - _to_logit(p_alt)
    return p_orig - p_alt

def _top_k(importance, /, *, k = None, percentile = None, abs_val = True):
    if (k is None) == (percentile is None):
        raise ValueError('Use k or percentile not both')
    key_fn = (lamda kv: abs(kv[1])) if abs_val else (lambda kv:kv[1])
    sorted_idx = sorted(importance.items(), key=key_fn, reverse=True)

    if k is None:
        k = max(1, int(np.ceil(len(sorted_idx) * percentile / 100)))
    k = min(k, len(sorted_idx))
    return {idx for idx, _ in sorted[:k]}

def compute_comprehensiveness(
        sample,
        importance,
        *,
        predict_fn,
        mask_fn,
        k = 10,
        percentile = None,
        use_logits = False):
    top = _top_k(importance, k=k, percentile=percentile)
    p_orig = float(predict_fn([sample])[0,0])
    samp_keep = mask_fn(sample, keep_idx)