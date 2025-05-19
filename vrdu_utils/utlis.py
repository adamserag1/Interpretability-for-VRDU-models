import torch
from __future__ import annotations

__all__ = ["to_fp16", "unnormalize_box"]

def to_fp16(batch, device):
    '''
    Move tensors in a batch to device, cast floating-point tensors to FP16 while leaving 
    non-floating types untouched
    '''
    out = {}
    for k, v in batch.items():
        if v.dtype.is_floating_point:      # only floats → half
            out[k] = v.half().to(device)
        else:                             # ints / bools stay as-is
            out[k] = v.to(device)
    return out

def unnormalize_box(bbox, width, height):
    '''
    Returns a bbox to its de-normalized state 
    '''
    return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
    ]
