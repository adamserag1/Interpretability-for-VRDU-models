from __future__ import annotations
import torch
from PIL import Image, ImageDraw
from matplotlib import cm, pyplot as plt
from vrdu_utils.module_types import *
from torch.utils.data import DataLoader, Dataset


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
    Convert absolute pixel coords (x0,y0,x1,y1) to 0-1000 scale *and*
    clamp every value into [0, 1000].  If the bbox already looks
    normalised, leave it unchanged.
    """
    if max(bbox) <= 1000:
        # assume already in 0-1000 layout units
        x0, y0, x1, y1 = bbox
    else:
        x0, y0, x1, y1 = bbox
        x0 = int(1000 * x0 / width)
        y0 = int(1000 * y0 / height)
        x1 = int(1000 * x1 / width)
        y1 = int(1000 * y1 / height)

    # hard clamp
    return [
        max(0, min(1000, x0)),
        max(0, min(1000, y0)),
        max(0, min(1000, x1)),
        max(0, min(1000, y1)),
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



def draw_lime_token_heatmap(
    image: Image.Image,
    words: Sequence[str],
    boxes: Sequence[Sequence[int]],
    weights: Mapping[str, float],
    *,
    alpha: float = 0.25,        # ← a bit lighter by default
    normalised: bool = False,
    outline: bool = False       # optional thin border
) -> Image.Image:
    """
    Return a **new** RGBA image with a transparent LIME overlay.

    Parameters
    ----------
    alpha : float
        Opacity of the coloured overlay (0-1).  0.25 keeps text legible.
    outline : bool
        If True, also draws a 1-px border around each coloured box.

    Notes
    -----
    • Works with pixel boxes or 0-1000 LayoutLM units (`normalised=True`).
    • Automatically centres the diverging colormap (blue ↔ white ↔ red).
    """
    if len(words) != len(boxes):
        raise ValueError("`words` and `boxes` must be the same length")

    base     = image.convert("RGBA")
    overlay  = Image.new("RGBA", base.size, (255, 255, 255, 0))   # fully transparent
    draw     = ImageDraw.Draw(overlay, mode="RGBA")

    cmap     = cm.get_cmap("coolwarm")
    max_abs  = max(abs(w) for w in weights.values()) or 1.0

    w_px, h_px = base.size

    for word, box in zip(words, boxes):
        if word not in weights:
            continue

        # convert to pixel coords if needed
        if normalised:
            x0, y0, x1, y1 = [
                int(coord / 1000 * dim)
                for coord, dim in zip(box, (w_px, h_px, w_px, h_px))
            ]
            box_px = [x0, y0, x1, y1]
        else:
            box_px = box

        # colour
        norm = weights[word] / max_abs      # –1 … +1
        r, g, b, _ = cmap(0.5 + 0.5 * norm)
        rgba = (int(r * 255), int(g * 255), int(b * 255), int(alpha * 255))

        draw.rectangle(box_px, fill=rgba)
        if outline:
            draw.rectangle(box_px, outline=rgba[:3] + (255,), width=1)

    # merge overlay with base
    return Image.alpha_composite(base, overlay)

def first_subtoken_of_word(word_idx: int, processor):
    """
    Returns a target_token_fn that finds the first sub-token position
    for `word_idx` (0-based) in *any* encoded batch of size 1.
    """
    def fn(enc):
        word_ids = processor.tokenizer.word_ids(batch_index=0)
        for pos, wid in enumerate(word_ids):
            if wid == word_idx:          # <-- first match is the first sub-token
                return pos
        raise ValueError(f"word_idx {word_idx} not found in batch")
    return fn

def row_to_docsample(ex):
    """
    ex is one row of the HF dataset (a plain dict).
    Returns a DocSample instance.
    """
    # -- guarantee types the encoders expect
    img = ex["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    label = ex["label"]

    w, h = img.size
    bbs = [normalize_bbox(b, w, h) for b in ex["bboxes"]]   # safety guard

    return DocSample(
        image   = img,
        words   = ex["words"],
        bboxes  = bbs,
        label   = int(ex["label"]),      # HF sometimes stores as np.int64
        ner_tags= ex.get("ner_tags"),    # may be absent for RVL
    )

class DocSampleDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __getitem__(self, idx):
        sample = self.ds[idx]
        try:
            label = sample["label"]
        except:
            label = 'FUNSD-SAMPLE'
        return DocSample(image=sample["image"], words=sample["words"], bboxes=sample["bboxes"], label=label, ner_tags=sample.get("ner_tags")), idx
        return row_to_docsample(self.ds[idx])

    def __len__(self):
        return len(self.ds)


def display_image_grid(images, titles, grid_size=(2, 2), figsize=(10, 10), main_title=None):
    if len(images) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Expected {grid_size[0] * grid_size[1]} images, got {len(images)}")

    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
    fig.suptitle(f'{main_title}', fontsize=20)
    for idx, (img, ax, title) in enumerate(zip(images, axes, titles)):
        # Convert PIL image to displayable format
        ax.imshow(img)
        ax.set_title(f"{title}")
        ax.axis('off')  # Hide axes for cleaner display
      # Adjust spacing between images
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    plt.show()