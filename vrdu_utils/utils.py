from __future__ import annotations
import torch
from PIL import Image, ImageDraw
from matplotlib import cm

__all__ = ["to_fp16", "normalize_bbox", "unnormalize_box"]

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

def normalize_bbox(bbox, width, height):
    """
    Convert absolute pixel coords to the 0-1000 LayoutLM/BROS convention.
    (0,0) is top-left of the page.
    """
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]

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


def draw_lime_token_heatmap(image,words,boxes,weights,*,alpha= 0.40,normalised = False):
    """
    """
    out = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(out, mode="RGBA")

    cmap = cm.get_cmap("coolwarm")
    max_abs = max(abs(w) for w in weights.values()) or 1.0

    w_px, h_px = out.size
    for word, box in zip(words, boxes):
        if word not in weights:
            continue

        # Convert to pixel coords if necessary
        if normalised:
            x0, y0, x1, y1 = [int(coord / 1000 * dim)
                              for coord, dim in zip(box, (w_px, h_px, w_px, h_px))]
            box_px = [x0, y0, x1, y1]
        else:
            box_px = box

        # Map weight to RGBA colour
        norm = weights[word] / max_abs         # → [-1, 1]
        r, g, b, _ = cmap(0.5 + 0.5 * norm)    # shift to [0,1] for cmap
        rgba = (int(r * 255), int(g * 255), int(b * 255), int(alpha * 255))

        draw.rectangle(box_px, fill=rgba)

    return out