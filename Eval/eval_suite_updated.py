import numpy as np
import torch
from copy import deepcopy
from PIL import Image, ImageFilter
from skimage.segmentation import slic
from tqdm.auto import tqdm

from datasets import load_dataset
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
    """
    Scalar confidence for *sample*.

    • If ``target_token_fn`` is None → document-level probability
      for the sample's ``label`` (classification).

    • Otherwise → token-level probability for ``target_label_id`` at the
      position returned by ``target_token_fn`` (NER).
    """
    enc = encode_fn([sample], device)
    if isinstance(enc, tuple):          # some encoders (NER) return (enc, _)
        enc = enc[0]

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits                    # (B, …)

    if target_token_fn is None:                         # classification
        probs = torch.softmax(logits, -1)
        return probs[0, sample.label].item()

    tok_idx = target_token_fn(enc)                      # token classification
    token_logits = logits[0][tok_idx]
    probs = torch.softmax(token_logits, -1)
    return probs[target_label_id].item()


def _select_top_k(explanation, k):
    """
    Return the *set* of the k most important features in ``explanation``.

    If k exceeds the number of available features, all features are returned.
    """
    items = sorted(explanation.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, min(k, len(items)))
    return {feat for feat, _ in items[:k]}


def _mask_text(words, feature_set, mask_token, keep):
    """
    Either *remove* or *keep only* the words in ``feature_set``.
    """
    out = []
    for w in words:
        in_set = w in feature_set
        if (in_set and not keep) or (not in_set and keep):
            out.append(mask_token)
        else:
            out.append(w)
    return out


def _mask_layout(bboxes, feature_set, image_size, keep):
    """
    Same logic as ``_mask_text`` but for bounding-boxes.
    """
    w, h = image_size
    full_box = [0, 0, w, h]

    out = []
    for box in bboxes:
        in_set = box in feature_set
        if (in_set and not keep) or (not in_set and keep):
            out.append(full_box)
        else:
            out.append(box)
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


def _mask_vision(image, feature_set, keep, blur_size, slic_kwargs):
    segments = _get_segments(image, **slic_kwargs)
    all_ids = set(np.unique(segments))
    to_blur = (all_ids - feature_set) if keep else feature_set
    return _blur_segments(image, segments, to_blur, blur_size)


# --------------------------------------------------------------------------- #
#  Metrics                                                                    #
# --------------------------------------------------------------------------- #

def evaluate_sample(sample, explanation, modality,
                    model, encode_fn, *,
                    top_k=1,                    # ← integer not fraction
                    device=None,
                    mask_token="[UNK]",
                    target_token_fn=None,
                    target_label_id=None,
                    blur_size=(64, 64),
                    slic_kwargs=None):
    """
    Return dict with ``comprehensiveness`` and ``sufficiency``.
    """
    slic_kwargs = slic_kwargs or dict(n_segments=200,
                                      compactness=20.0, sigma=1.0)
    device = torch.device(device or ("cuda"
                                     if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    feature_set = _select_top_k(explanation, top_k)
    original = _predict(model, encode_fn, device, sample,
                        target_token_fn, target_label_id)

    # ---------------------  comprehensiveness (mask top-k) ------------------ #
    if modality == "text":
        words = _mask_text(sample.words, feature_set, mask_token, keep=False)
        comp_sample = DocSample(sample.image, words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "layout":
        boxes = _mask_layout(sample.bboxes, feature_set,
                             sample.image.size, keep=False)
        comp_sample = DocSample(sample.image, sample.words, boxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "vision":
        img = _mask_vision(sample.image, feature_set, keep=False,
                           blur_size=blur_size, slic_kwargs=slic_kwargs)
        comp_sample = DocSample(img, sample.words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)
    else:
        raise ValueError("modality must be 'text', 'layout' or 'vision'")

    comp_prob = _predict(model, encode_fn, device, comp_sample,
                         target_token_fn, target_label_id)
    comprehensiveness = original - comp_prob

    # -----------------------  sufficiency (keep top-k) ---------------------- #
    if modality == "text":
        words = _mask_text(sample.words, feature_set, mask_token, keep=True)
        suff_sample = DocSample(sample.image, words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "layout":
        boxes = _mask_layout(sample.bboxes, feature_set,
                             sample.image.size, keep=True)
        suff_sample = DocSample(sample.image, sample.words, boxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    elif modality == "vision":
        img = _mask_vision(sample.image, feature_set, keep=True,
                           blur_size=blur_size, slic_kwargs=slic_kwargs)
        suff_sample = DocSample(img, sample.words, sample.bboxes,
                                label=sample.label, ner_tags=sample.ner_tags)

    suff_prob = _predict(model, encode_fn, device, suff_sample,
                         target_token_fn, target_label_id)
    sufficiency = original - suff_prob

    return {"comprehensiveness": comprehensiveness,
            "sufficiency": sufficiency}


def evaluate_batch(samples, explanations, modality,
                   model, encode_fn, *,
                   top_k=1,
                   device=None,
                   mask_token="[UNK]",
                   target_token_fn=None,
                   target_label_id=None,
                   blur_size=(64, 64),
                   slic_kwargs=None):
    """
    Evaluate `samples` ↔ `explanations` in a vectorised loop.
    """
    out = []
    for s, e in tqdm(list(zip(samples, explanations)),
                     desc="Evaluating Fidelity"):
        out.append(
            evaluate_sample(s, e, modality, model, encode_fn,
                            top_k=top_k, device=device, mask_token=mask_token,
                            target_token_fn=target_token_fn,
                            target_label_id=target_label_id,
                            blur_size=blur_size,
                            slic_kwargs=slic_kwargs)
        )
    return out


# --------------------------------------------------------------------------- #
#  Convenience wrappers (unchanged in spirit, updated param names)            #
# --------------------------------------------------------------------------- #

class ModelWrapper:
    """
    Provides ``encode_fn`` plus the underlying ``model``.
    """
    def __init__(self, model, processor,
                 model_type, task_type, device=None):
        self.model = model.eval()
        self.processor = processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ner = task_type.lower() == "ner"
        if model_type.lower() == "layoutlmv3":
            self.encode_fn = make_layoutlmv3_encoder(processor, ner)
        elif model_type.lower() == "bros":
            self.encode_fn = make_bros_encoder(processor, ner)
        else:
            raise ValueError(f"unknown model_type {model_type!r}")


class EvaluationSuite:
    """
    High-level orchestrator.  Public `evaluate_fidelity` now accepts *top_k*.
    """
    def __init__(self, model_name, model_type, task_type, *,
                 device=None, blur_size=(64, 64), slic_kwargs=None,
                 mask_token="[UNK]"):
        self.task_type = task_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.blur_size = blur_size
        self.slic_kwargs = slic_kwargs or dict(n_segments=200,
                                               compactness=20.0, sigma=1.0)
        self.mask_token = mask_token

        if self.task_type == "document_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif self.task_type == "ner":
            model = AutoModelForTokenClassification.from_pretrained(model_name)
        else:
            raise ValueError(f"unsupported task_type {task_type!r}")

        processor = AutoProcessor.from_pretrained(model_name)
        self.wrapper = ModelWrapper(model, processor, model_type,
                                    task_type, self.device)

    # (Dataset loader unchanged …)

    def evaluate_fidelity(self, samples, explanations, modality, *,
                          top_k=1, target_token_fn=None, target_label_id=None):
        return evaluate_batch(samples, explanations, modality,
                              self.wrapper.model, self.wrapper.encode_fn,
                              top_k=top_k,
                              device=self.device,
                              mask_token=self.mask_token,
                              target_token_fn=target_token_fn,
                              target_label_id=target_label_id,
                              blur_size=self.blur_size,
                              slic_kwargs=self.slic_kwargs)
