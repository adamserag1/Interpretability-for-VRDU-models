"""
Evaluation + AOPC curves for VRDU interpretability
=================================================

Two new public helpers
----------------------
• `compute_aopc_curves(...)`
    – Returns the AOPC values *per k* for every {model, explainer} pair
      (keys: `"BROS lime"`, `"LLMV3 lime"`, `"LLMV3 shap"`).

• `plot_aopc_curves(aopc_dict, modality, title=None)`
    – Matplotlib plot with one line per pair and an automatic legend.

Everything else (fidelity metrics, encoders, wrappers) is unchanged from the
previous version, except for internal refactors to avoid code-duplication.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from PIL import Image, ImageFilter
from skimage.segmentation import slic
from tqdm.auto import tqdm
from typing import Dict, List, Mapping, Sequence

from transformers import (
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from vrdu_utils.module_types import DocSample
from vrdu_utils.encoders import make_layoutlmv3_encoder, make_bros_encoder


# --------------------------------------------------------------------------- #
#  Low-level helpers                                                          #
# --------------------------------------------------------------------------- #

def _predict(model, encode_fn, device, sample,
             target_token_fn=None, target_label_id=None):
    enc = encode_fn([sample], device)
    if isinstance(enc, tuple):
        enc = enc[0]

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    if target_token_fn is None:               # classification
        probs = torch.softmax(logits, -1)
        return probs[0, sample.label].item()

    tok_idx = target_token_fn(enc)            # token-classification
    token_logits = logits[0][tok_idx]
    probs = torch.softmax(token_logits, -1)
    return probs[target_label_id].item()


def _top_k_list(explanation: Mapping, k: int) -> List:
    """
    Return the *ordered list* of the k most-important features
    (most→least importance).  Caps at the number of available features.
    """
    k = max(1, k)
    items = sorted(explanation.items(),
                   key=lambda kv: kv[1], reverse=True)[:k]
    return [feat for feat, _ in items]


def _mask_text(words, feat_set, mask_token, keep):
    out = []
    for w in words:
        in_set = w in feat_set
        if (in_set and not keep) or (not in_set and keep):
            out.append(mask_token)
        else:
            out.append(w)
    return out


def _mask_layout(bboxes, feat_set, image_size, keep):
    w, h = image_size
    full_box = [0, 0, w, h]
    out = []
    for b in bboxes:
        in_set = b in feat_set
        out.append(full_box if (in_set and not keep) or (not in_set and keep)
                   else b)
    return out


# ------------------------------  vision ------------------------------------ #

def _get_segments(image, n_segments=200, compactness=20.0, sigma=1.0):
    arr = np.asarray(image.convert("RGB"))
    return slic(arr, n_segments=n_segments,
                compactness=compactness, sigma=sigma)


def _blur_segments(image, segments, seg_ids, blur_size):
    arr = np.asarray(image.convert("RGB"))
    radius = max(blur_size) // 2 or 1
    blurred = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius))
    blurred_arr = np.asarray(blurred)
    out = arr.copy()
    for sid in seg_ids:
        mask = segments == sid
        out[mask] = blurred_arr[mask]
    return Image.fromarray(out)


def _mask_vision(image, feat_set, keep, blur_size, slic_kwargs):
    segments = _get_segments(image, **slic_kwargs)
    all_ids = set(np.unique(segments))
    to_blur = (all_ids - feat_set) if keep else feat_set
    return _blur_segments(image, segments, to_blur, blur_size)


# --------------------------------------------------------------------------- #
#  Fidelity metrics (unchanged signature)                                     #
# --------------------------------------------------------------------------- #

def evaluate_sample(sample: DocSample,
                    explanation: Mapping,
                    modality: str,
                    model, encode_fn, *,
                    top_k: int = 1,
                    device=None,
                    mask_token: str = "[UNK]",
                    target_token_fn=None,
                    target_label_id=None,
                    blur_size=(64, 64),
                    slic_kwargs=None):

    slic_kwargs = slic_kwargs or dict(n_segments=200,
                                      compactness=20.0, sigma=1.0)
    device = torch.device(device or ("cuda"
                                     if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    feat_list = _top_k_list(explanation, top_k)
    feat_set = set(feat_list)

    original = _predict(model, encode_fn, device, sample,
                        target_token_fn, target_label_id)

    # comprehensiveness ------------------------------------------------------ #
    if modality == "text":
        words = _mask_text(sample.words, feat_set, mask_token, keep=False)
        comp_sample = DocSample(sample.image, words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "layout":
        boxes = _mask_layout(sample.bboxes, feat_set,
                             sample.image.size, keep=False)
        comp_sample = DocSample(sample.image, sample.words, boxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "vision":
        img = _mask_vision(sample.image, feat_set, keep=False,
                           blur_size=blur_size, slic_kwargs=slic_kwargs)
        comp_sample = DocSample(img, sample.words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)
    else:
        raise ValueError("modality must be 'text', 'layout' or 'vision'")

    comp_prob = _predict(model, encode_fn, device, comp_sample,
                         target_token_fn, target_label_id)
    comprehensiveness = original - comp_prob

    # sufficiency ------------------------------------------------------------ #
    if modality == "text":
        words = _mask_text(sample.words, feat_set, mask_token, keep=True)
        suff_sample = DocSample(sample.image, words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "layout":
        boxes = _mask_layout(sample.bboxes, feat_set,
                             sample.image.size, keep=True)
        suff_sample = DocSample(sample.image, sample.words, boxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "vision":
        img = _mask_vision(sample.image, feat_set, keep=True,
                           blur_size=blur_size, slic_kwargs=slic_kwargs)
        suff_sample = DocSample(img, sample.words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    suff_prob = _predict(model, encode_fn, device, suff_sample,
                         target_token_fn, target_label_id)
    sufficiency = original - suff_prob

    return {"comprehensiveness": comprehensiveness,
            "sufficiency": sufficiency,
            "original_prob": original}


# --------------------------------------------------------------------------- #
#  AOPC                                                                       #
# --------------------------------------------------------------------------- #

def _aopc_single(sample: DocSample,
                 explanation: Mapping,
                 modality: str,
                 model, encode_fn, *,
                 max_k: int,
                 device=None,
                 mask_token="[UNK]",
                 target_token_fn=None,
                 target_label_id=None,
                 blur_size=(64, 64),
                 slic_kwargs=None):
    """
    Returns an array length `max_k` with cumulative probability drop
    after masking 1,2,…,k features (AOPC numerator per step).
    """
    slic_kwargs = slic_kwargs or {}
    device = torch.device(device or ("cuda"
                                     if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    original = _predict(model, encode_fn, device, sample,
                        target_token_fn, target_label_id)

    aopc = []
    feat_list = _top_k_list(explanation, max_k)

    for k in range(1, max_k + 1):
        remove_set = set(feat_list[:k])

        if modality == "text":
            words = _mask_text(sample.words, remove_set, mask_token, keep=False)
            pert = DocSample(sample.image, words, sample.bboxes,
                             ner_tags=sample.ner_tags, label=sample.label)

        elif modality == "layout":
            boxes = _mask_layout(sample.bboxes, remove_set,
                                 sample.image.size, keep=False)
            pert = DocSample(sample.image, sample.words, boxes,
                             ner_tags=sample.ner_tags, label=sample.label)

        elif modality == "vision":
            img = _mask_vision(sample.image, remove_set, keep=False,
                               blur_size=blur_size, slic_kwargs=slic_kwargs)
            pert = DocSample(img, sample.words, sample.bboxes,
                             ner_tags=sample.ner_tags, label=sample.label)
        else:
            raise ValueError(modality)

        prob = _predict(model, encode_fn, device, pert,
                        target_token_fn, target_label_id)
        aopc.append(original - prob)

    return np.array(aopc)


def compute_aopc_curves(samples: Sequence[DocSample],
                        explanations_dict: Dict[str, Sequence[Mapping]],
                        modality: str,
                        model_wrappers: Dict[str, 'ModelWrapper'],
                        *,
                        max_k: int = 10,
                        blur_size=(64, 64),
                        slic_kwargs=None,
                        mask_token="[UNK]") -> Dict[str, np.ndarray]:
    """
    Calculate AOPC (Average-over-k) curves for every key in `explanations_dict`.

    Parameters
    ----------
    explanations_dict : {"BROS lime": …, "LLMV3 lime": …, "LLMV3 shap": …}
        Each entry is a *list* of explanation dicts aligned with `samples`.
    model_wrappers : same keys → ModelWrapper containing `.model` & `.encode_fn`
    Returns
    -------
    dict key → np.ndarray shape (max_k,)
        Mean cumulative drop at each k position.
    """
    curves = {}
    for key, exps in explanations_dict.items():
        mw = model_wrappers[key]
        drops = []
        for samp, exp in tqdm(zip(samples, exps),
                              total=len(samples),
                              desc=f"AOPC ({key})"):
            drops.append(
                _aopc_single(samp, exp, modality,
                             mw.model, mw.encode_fn,
                             max_k=max_k, mask_token=mask_token,
                             blur_size=blur_size, slic_kwargs=slic_kwargs)
            )
        curves[key] = np.mean(np.vstack(drops), axis=0)
    return curves


def plot_aopc_curves(aopc_dict: Dict[str, np.ndarray],
                     modality: str,
                     title: str | None = None):
    """
    Simple matplotlib line plot with legend.  Colours chosen automatically.
    """
    plt.figure()
    for label, curve in aopc_dict.items():
        ks = np.arange(1, len(curve) + 1)
        plt.plot(ks, curve, label=label)
    plt.xlabel("k masked features")
    plt.ylabel("Δ probability (avg)")
    plt.title(title or f"AOPC – {modality}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
#  Wrappers (unchanged from previous drop-in)                                 #
# --------------------------------------------------------------------------- #

class ModelWrapper:
    """
    Holds (`model`, `encode_fn`) for a given backbone, ready for AOPC.
    """
    def __init__(self, model, processor,
                 model_type, task_type, device=None):
        self.model = model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ner = task_type.lower() == "ner"
        if model_type.lower() == "layoutlmv3":
            self.encode_fn = make_layoutlmv3_encoder(processor, ner)
        elif model_type.lower() == "bros":
            self.encode_fn = make_bros_encoder(processor, ner)
        else:
            raise ValueError(f"unknown model_type {model_type!r}")


# ------------------------------------------------------- usage snippet ------
#
# wrappers = {
#     "BROS lime"   : ModelWrapper(bros_model , bros_proc , "bros"      , "document_classification"),
#     "LLMV3 lime"  : ModelWrapper(llm_model  , llm_proc  , "layoutlmv3", "document_classification"),
#     "LLMV3 shap"  : ModelWrapper(llm_model  , llm_proc  , "layoutlmv3", "document_classification"),
# }
#
# aopc = compute_aopc_curves(samples,
#                            explanations_dict,   # same keys as wrappers
#                            modality="text",
#                            model_wrappers=wrappers,
#                            max_k=10)
#
# plot_aopc_curves(aopc, modality="text")
#
# --------------------------------------------------------------------------- #
