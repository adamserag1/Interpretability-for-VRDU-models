
Evaluation suite for VRDU interpretability models.
import numpy as np
import torch
from copy import deepcopy
from PIL import Image, ImageFilter
from skimage.segmentation import slic
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSequenceClassification, AutoModelForTokenClassification

from vrdu_utils.module_types import DocSample
from vrdu_utils.encoders import make_layoutlmv3_encoder, make_bros_encoder


def _predict(model, encode_fn, device, sample, target_token_fn=None, target_label_id=None):
    """
    Return the model's probability for a given sample.

    If ``target_token_fn`` is ``None`` this function assumes a document
    classification task and returns the probability of the sample's label.
    Otherwise it treats the task as token classification (e.g. NER) and
    returns the probability of ``target_label_id`` for the token identified
    by ``target_token_fn``.
    """
    # Encode the sample; some encoders for NER return a tuple
    enc = encode_fn([sample], device)
    if isinstance(enc, tuple):
        enc = enc[0]
    # Move tensors to the appropriate device
    for k, v in enc.items():
        enc[k] = v.to(device)

    with torch.no_grad():
        logits = model(**enc).logits
    if target_token_fn is None:
        probs = torch.softmax(logits, dim=-1)
        return probs[0, sample.label].item()
    # token classification: select the token specified by target_token_fn
    tok_idx = target_token_fn(enc)
    token_logits = logits[0][tok_idx]
    token_probs = torch.softmax(token_logits, -1)
    return token_probs[target_label_id].item()


def _top_k_features(explanation, fraction):
    """
    Given an explanation mapping features to importance scores, return a set
    containing the top‑k features.  ``fraction`` determines the proportion of
    features to select; at least one feature is always returned.
    """
    items = list(explanation.items())
    items.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(len(items) * fraction))
    return {feature for feature, _ in items[:k]}


def _mask_text(words, remove_set, mask_token, keep):
    """
    Mask or keep selected words.  When ``keep`` is ``False`` every word in
    ``remove_set`` is replaced by ``mask_token``; otherwise only words in
    ``remove_set`` are kept and all others are masked.  A new list of words
    is returned.
    """
    out = []
    for w in words:
        in_set = w in remove_set
        # XOR ensures we mask when appropriate for comprehensiveness or sufficiency
        if (in_set and not keep) or (not in_set and keep):
            out.append(mask_token)
        else:
            out.append(w)
    return out


def _mask_layout(bboxes, remove_set, image_size, keep):
    """
    Mask or keep selected bounding boxes.  For boxes in the ``remove_set``
    (or outside it when ``keep`` is true) the entire page bounding box is
    substituted.  A new list of boxes is returned.
    """
    w, h = image_size
    full_box = [0, 0, w, h]
    out = []
    for box in bboxes:
        in_set = box in remove_set
        if (in_set and not keep) or (not in_set and keep):
            out.append(full_box)
        else:
            out.append(box)
    return out


def _get_segments(image, n_segments=200, compactness=20.0, sigma=1.0):
    """
    Compute a SLIC segmentation for an RGB image.  Returns a 2‑D array of
    integer segment labels.
    """
    arr = np.asarray(image.convert('RGB'))
    return slic(arr, n_segments=n_segments, compactness=compactness, sigma=sigma)


def _blur_segments(image, segments, seg_ids, blur_size):
    """
    Given a segmentation map and a set of segment identifiers, return a new
    image where the specified segments have been blurred using a Gaussian
    filter.  ``blur_size`` is a tuple whose largest element is used as the
    blur radius.
    """
    arr = np.asarray(image.convert('RGB'))
    # Precompute a blurred version of the whole image
    radius = max(blur_size) // 2 or 1
    blurred = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius))
    blurred_arr = np.asarray(blurred)
    out = arr.copy()
    for seg_id in seg_ids:
        mask = segments == seg_id
        out[mask] = blurred_arr[mask]
    return Image.fromarray(out)


def _mask_vision(image, remove_set, keep, blur_size, slic_kwargs):
    """
    Mask or keep selected image segments.  When ``keep`` is ``False`` the
    segments listed in ``remove_set`` are blurred; when ``keep`` is ``True``
    all other segments are blurred.  Segmentation is computed per call using
    the supplied ``slic_kwargs``.
    """
    segments = _get_segments(image, **slic_kwargs)
    all_ids = set(np.unique(segments))
    if keep:
        to_blur = all_ids - set(remove_set)
    else:
        to_blur = set(remove_set)
    return _blur_segments(image, segments, to_blur, blur_size)


def evaluate_sample(sample, explanation, modality, model, encode_fn, device=None,
                    mask_token="[UNK]", top_k_fraction=0.2, target_token_fn=None,
                    target_label_id=None, blur_size=(64, 64), slic_kwargs=None):
    """
    Compute comprehensiveness and sufficiency for a single sample and its
    explanation.  The modality determines which part of the sample to
    perturb.  ``top_k_fraction`` controls how many features are masked.
    """
    if slic_kwargs is None:
        slic_kwargs = dict(n_segments=200, compactness=20.0, sigma=1.0)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()

    # Determine which features to perturb
    remove_set = _top_k_features(explanation, top_k_fraction)

    # Compute original probability
    original_prob = _predict(model, encode_fn, device, sample, target_token_fn, target_label_id)

    # Comprehensiveness: mask the top features
    if modality == 'text':
        words = _mask_text(sample.words, remove_set, mask_token, keep=False)
        perturbed = DocSample(sample.image, words, sample.bboxes, label=sample.label, ner_tags=sample.ner_tags)
    elif modality == 'layout':
        boxes = _mask_layout(sample.bboxes, remove_set, sample.image.size, keep=False)
        perturbed = DocSample(sample.image, sample.words, boxes, label=sample.label, ner_tags=sample.ner_tags)
    elif modality == 'vision':
        image = _mask_vision(sample.image, remove_set, keep=False, blur_size=blur_size, slic_kwargs=slic_kwargs)
        perturbed = DocSample(image, sample.words, sample.bboxes, label=sample.label, ner_tags=sample.ner_tags)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    perturbed_prob = _predict(model, encode_fn, device, perturbed, target_token_fn, target_label_id)
    comprehensiveness = original_prob - perturbed_prob

    # Sufficiency: keep only the top features and mask the rest
    if modality == 'text':
        words = _mask_text(sample.words, remove_set, mask_token, keep=True)
        suff_perturbed = DocSample(sample.image, words, sample.bboxes, label=sample.label, ner_tags=sample.ner_tags)
    elif modality == 'layout':
        boxes = _mask_layout(sample.bboxes, remove_set, sample.image.size, keep=True)
        suff_perturbed = DocSample(sample.image, sample.words, boxes, label=sample.label, ner_tags=sample.ner_tags)
    elif modality == 'vision':
        image = _mask_vision(sample.image, remove_set, keep=True, blur_size=blur_size, slic_kwargs=slic_kwargs)
        suff_perturbed = DocSample(image, sample.words, sample.bboxes, label=sample.label, ner_tags=sample.ner_tags)
    suff_prob = _predict(model, encode_fn, device, suff_perturbed, target_token_fn, target_label_id)
    sufficiency = original_prob - suff_prob

    return {'comprehensiveness': comprehensiveness, 'sufficiency': sufficiency}


def evaluate_batch(samples, explanations, modality, model, encode_fn, device=None,
                   mask_token="[UNK]", top_k_fraction=0.2, target_token_fn=None,
                   target_label_id=None, blur_size=(64, 64), slic_kwargs=None):
    """
    Compute fidelity metrics for a batch of samples and their corresponding
    explanations.  Returns a list of dictionaries.
    """
    results = []
    for sample, explanation in tqdm(list(zip(samples, explanations)), desc="Evaluating Fidelity"):
        metrics = evaluate_sample(sample, explanation, modality, model, encode_fn,
                                  device=device, mask_token=mask_token,
                                  top_k_fraction=top_k_fraction,
                                  target_token_fn=target_token_fn,
                                  target_label_id=target_label_id,
                                  blur_size=blur_size,
                                  slic_kwargs=slic_kwargs)
        results.append(metrics)
    return results


class ModelWrapper:
    """
    A thin wrapper around a model and its processor/tokeniser.  It
    provides a consistent interface for encoding ``DocSample`` objects and
    exposes the underlying model and its type.  The ``model_type`` can be
    either ``"layoutlmv3"`` or ``"bros"``.
    """
    def __init__(self, model, processor, model_type, task_type, device=None):
        self.model = model.eval()
        self.processor = processor
        self.model_type = model_type
        self.task_type = task_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Select appropriate encoder
        if model_type.lower() == 'layoutlmv3':
            ner = task_type.lower() == 'ner'
            self.encode_fn = make_layoutlmv3_encoder(processor, ner)
        elif model_type.lower() == 'bros':
            ner = task_type.lower() == 'ner'
            self.encode_fn = make_bros_encoder(processor, ner)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class EvaluationSuite:
    """
    High‑level helper for loading models, preparing datasets and evaluating
    fidelity metrics.  It supports document classification and NER tasks on
    the FUNSD and RVL‑CDIP datasets.  The ``evaluate_fidelity`` method
    delegates to ``evaluate_batch`` defined above.
    """
    def __init__(self, model_name, model_type, task_type, device=None,
                 blur_size=(64, 64), slic_kwargs=None, mask_token="[UNK]"):
        self.model_name = model_name
        self.model_type = model_type
        self.task_type = task_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.blur_size = blur_size
        self.slic_kwargs = slic_kwargs or dict(n_segments=200, compactness=20.0, sigma=1.0)
        self.mask_token = mask_token

        # Load model and processor based on task
        if task_type.lower() == 'document_classification':
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task_type.lower() == 'ner':
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.wrapper = ModelWrapper(self.model, self.processor, model_type, task_type, self.device)

    def load_dataset(self, dataset_name, split='test'):
        """
        Load and prepare a dataset for the specified task.  For document
        classification this method currently assumes a dummy dataset; for
        NER it loads the FUNSD dataset from the Hub.  The return value is a
        list of ``DocSample`` instances.
        """
        if dataset_name.lower() == 'funsd' and self.task_type.lower() == 'ner':
            dataset = load_dataset('nielsr/funsd', split=split)
            processed = []
            for entry in dataset:
                words = [w for w in entry['text']]
                bboxes = [box for box in entry['box']]
                image = entry['image']
                tags = entry['ner_tags']
                processed.append(DocSample(image=image, words=words, bboxes=bboxes,
                                           ner_tags=tags, label=None))
            return processed
        if dataset_name.lower() == 'rvl_cdip' and self.task_type.lower() == 'document_classification':
            # For RVL‑CDIP assume the user provides their own list of DocSample objects.
            raise RuntimeError('Please provide your RVL‑CDIP samples as a list of DocSample objects.')
        raise ValueError(f"Unsupported dataset {dataset_name} for task {self.task_type}")

    def evaluate_fidelity(self, samples, explanations, modality, top_k_fraction=0.2,
                          target_token_fn=None, target_label_id=None):
        """
        Evaluate comprehensiveness and sufficiency for a batch of samples and
        explanations.  The modality must be one of ``"text"``, ``"layout"``
        or ``"vision"``.  Optional arguments ``target_token_fn`` and
        ``target_label_id`` are used for NER tasks.
        """
        return evaluate_batch(samples, explanations, modality,
                              self.wrapper.model, self.wrapper.encode_fn,
                              device=self.device,
                              mask_token=self.mask_token,
                              top_k_fraction=top_k_fraction,
                              target_token_fn=target_token_fn,
                              target_label_id=target_label_id,
                              blur_size=self.blur_size,
                              slic_kwargs=self.slic_kwargs)