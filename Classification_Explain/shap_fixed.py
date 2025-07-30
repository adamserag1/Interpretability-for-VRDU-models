'''
SHAP explanation utilities for VRDU models.

This module provides classes to compute SHAP (SHapley Additive
explanations) values for multimodal document classifiers such as
LayoutLMv3 and BROS.  The original implementation shipped with this
repository returned class probabilities when making predictions and then
attempted to explain those probabilities directly.  According to the
SHAP documentation, it is often better to work in a log–odds (logit)
space when explaining classification models so that contributions add
linearly【697382921012888†L2521-L2528】.  The official SHAP examples
illustrating how to build a custom explainer for text classification
explicitly transform predicted probabilities into log‑odds prior to
passing them back to SHAP【70757593867465†L87-L97】.  This module
implements the same strategy: predictions are converted to
log‑odds so that the resulting SHAP values are meaningful and sum to the
difference between the base value (the output when all features are
masked) and the logit of the full prediction.

If you need to explain the text modality for a DocSample, use
``SHAPTextExplainer``.  For vision‑only explanations, use
``SHAPVisionExplainer``.  ``SHAPLayoutExplainer`` extends the text
explainer by additionally masking bounding boxes when tokens are
removed.

Example usage:

.. code-block:: python

    model.eval()
    explainer = SHAPTextExplainer(model, encode_fn, tokenizer)
    sample = DocSample(image=img, words=words, bboxes=bboxes,
                       ner_tags=None, label=label)
    shap_values = explainer.explain(sample)

"""
'''
from __future__ import annotations

import numpy as np
import torch
import shap
from vrdu_utils.module_types import DocSample
from PIL import Image
from functools import wraps
from transformers import LayoutLMv3TokenizerFast


def make_layoutlmv3_tokenizer_wrapper(tkn: LayoutLMv3TokenizerFast,
                                      dummy_box=(0, 0, 0, 0)):
    """Return a callable that always feeds LayoutLM‑v3 a
    ``(words, boxes, is_split_into_words=True)`` triple — even for the
    empty‑string probe used by SHAP.

    SHAP's ``Text`` masker sometimes calls a tokenizer with an empty
    string.  HuggingFace's LayoutLMv3 tokenizer expects a list of
    words and associated bounding boxes.  This wrapper ensures that
    call signatures remain consistent by converting strings into lists
    of space‑separated tokens and supplying dummy boxes when
    necessary.
    """

    def _to_words(x):
        # already a list/tuple => keep; else split on whitespace
        return list(x) if isinstance(x, (list, tuple)) else x.split()

    @wraps(tkn)
    def wrapped(texts, **kwargs):
        # ---------------- single example ----------------
        if isinstance(texts, str):
            words = _to_words(texts)  # [] for ""
            if "boxes" not in kwargs:
                kwargs["boxes"] = [dummy_box] * len(words)
            return tkn(words, **kwargs)

        # ---------------- batch -------------------------
        batch_words = [_to_words(s) for s in texts]  # list[list[str]]
        if "boxes" not in kwargs:
            kwargs["boxes"] = [[dummy_box] * len(seq) for seq in batch_words]
        return tkn(batch_words, **kwargs)

    return wrapped


# Delimiter for joining and splitting sentinel‑delimited documents.
DELIMITER = " "


def safe_split(doc_str: str) -> list[str]:
    """Split a sentinel‑delimited document string into tokens."""
    return doc_str.split(DELIMITER)


def sentinel_tokenizer(s: str, return_offsets_mapping: bool = True, **kw):
    """Custom tokenizer for SHAP ``Text`` masker using a sentinel delimiter.

    SHAP's ``Text`` masker expects a minimal subset of the HuggingFace
    tokenizer API that returns an ``input_ids`` list and an
    ``offset_mapping`` when ``return_offsets_mapping=True``.  This
    function splits on ``DELIMITER`` and computes character offsets for
    each token so that SHAP can map SHAP values back to substrings.
    """
    toks = [t for t in s.split(DELIMITER) if t]
    ids = list(range(len(toks)))
    if return_offsets_mapping:
        offs = []
        pos = 0
        for t in toks:
            offs.append((pos, pos + len(t)))
            pos += len(t) + len(DELIMITER)
        return {"input_ids": ids, "offset_mapping": offs}
    else:
        return {"input_ids": ids}


class BaseShapExplainer:
    """
    Base class for all SHAP explainers in this module.

    Subclasses should implement a ``_make_predict_fn`` method that
    accepts a ``DocSample`` and returns a function compatible with
    SHAP's ``Explainer``.  This base class handles common
    infrastructure such as device management and conversion of model
    outputs into log‑odds for classification problems.
    """

    def __init__(self, model, encode_fn, algorithm: str = 'partition', device: str | None = None):
        self.model = model.eval()
        self.encode_fn = encode_fn
        # select device; fallback to CUDA if available else CPU
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        # gather class names if the model exposes id2label (HuggingFace convention)
        self.class_names = [v for k, v in sorted(model.config.id2label.items())] if hasattr(model.config, "id2label") else []
        self.algorithm = algorithm
        # placeholder for the shap.Explainer instance; subclasses may cache it
        self.explainer: shap.Explainer | None = None

    def _encode(self, samples: list[DocSample] | DocSample):
        """Encode samples using the provided encode function."""
        return self.encode_fn(samples, self.device)

    @torch.no_grad()
    def _predict(self, samples: list[DocSample], temp: float = 1.0) -> np.ndarray:
        """
        Predict log‑odds for a batch of ``DocSample`` instances.

        The original implementation returned class probabilities via
        ``torch.softmax``.  However, explaining probabilities directly
        makes the SHAP values difficult to interpret because the
        relationship between features and probabilities is non‑linear.
        Following the SHAP documentation, we instead convert model
        outputs into log‑odds (logits of the probability for each
        class).  Log‑odds add linearly, so the sum of the SHAP values
        across features equals the difference between the logit of the
        full prediction and the base value when all features are
        masked【697382921012888†L2521-L2528】.

        Parameters
        ----------
        samples : list[DocSample]
            A list of samples to encode and predict.
        temp : float, optional
            Temperature scaling factor.  Unused in this implementation
            but retained for API compatibility.

        Returns
        -------
        np.ndarray
            Array of shape (N, C) containing log‑odds for each sample
            and class.
        """
        # encode the batch
        encoded = self._encode(samples)
        try:
            logits = self.model(**encoded).logits  # HuggingFace models expose .logits
        except Exception:
            # some models return a dict
            logits = self.model(**encoded)['logits']

        # compute probabilities and then convert to log‑odds
        probs = torch.softmax(logits, dim=-1)
        # add a small epsilon to avoid division by zero when p≈1
        eps = 1e-6
        log_odds = torch.log(probs / (1.0 - probs + eps))
        return log_odds.cpu().numpy()

    def explain(self, data, **shap_kwargs):
        """
        Explain a single input via an existing explainer instance.

        Subclasses are responsible for creating ``self.explainer``.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialised. Did you call the subclass constructor correctly?")
        return self.explainer(data, **shap_kwargs)

    def explain_batch(self, samples: list[DocSample], **kwargs):
        """
        Explain a batch of samples by iterating through them.
        """
        return [self.explain(s, **kwargs) for s in samples]


class SHAPTextExplainer(BaseShapExplainer):
    """
    SHAP explainer for the text modality.

    This class mirrors the structure of a LimeText explainer but uses
    SHAP's ``Partition`` algorithm by default.  It builds a masking
    function that perturbs the words of a ``DocSample`` by replacing
    them with a mask token and then calls the underlying model to
    obtain predictions in log‑odds space.  The resulting SHAP values
    represent the contribution of each word toward each output class.
    """

    def __init__(
        self,
        model,
        encode_fn,
        tokenizer,
        mask_token: str = "[UNK]",
        device: str | None = None,
        batch_size: int = 16,
        algorithm: str = 'partition'
    ):
        super().__init__(model, encode_fn, algorithm, device)
        self.mask_token = mask_token
        self.batch_size = batch_size
        self.algorithm = algorithm
        # if a LayoutLMv3 tokenizer is passed, wrap it to handle empty strings
        if isinstance(tokenizer, LayoutLMv3TokenizerFast):
            self.tokenizer = make_layoutlmv3_tokenizer_wrapper(tokenizer)
        else:
            self.tokenizer = tokenizer

    def _batched_predict(self, samples: list[DocSample]) -> np.ndarray:
        """
        Batch predictions over a list of ``DocSample`` instances to avoid
        out‑of‑memory errors.
        """
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i: i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, sample: DocSample, align_boxes: bool = False):
        """
        Construct a function mapping perturbed text strings to model
        predictions in log‑odds space.

        The returned function accepts a list of perturbed sentences and
        reconstructs ``DocSample`` instances by pairing each perturbed
        sentence with bounding boxes from the original sample.  If
        ``align_boxes`` is True, masked tokens collapse their bounding
        boxes to the full page so that the contribution of a missing
        token does not depend on its original region.
        """
        # full page dimensions for dummy bounding boxes
        W, H = sample.image.size
        dummy_box = [0, 0, W, H]

        def fn(perturbed_texts: list[str]) -> np.ndarray:
            ds_batch: list[DocSample] = []
            for sent in perturbed_texts:
                words = sent.split()  # SHAP inserts spaces between tokens
                boxes: list[list[int]] = []
                for i, w in enumerate(words):
                    if i < len(sample.bboxes):
                        # keep original box or collapse it, depending on align_boxes
                        if align_boxes and w == self.mask_token:
                            boxes.append(dummy_box)
                        else:
                            boxes.append(sample.bboxes[i])
                    else:
                        # SHAP produced a new word we never had – give it a dummy box
                        boxes.append(dummy_box)
                ds_batch.append(
                    DocSample(
                        image=sample.image,
                        words=words,
                        bboxes=boxes,
                        ner_tags=sample.ner_tags,
                        label=sample.label,
                    )
                )
            return self._batched_predict(ds_batch)

        return fn

    def explain(
        self,
        sample: DocSample,
        align_boxes: bool = False,
        num_samples: int = 2000,
    ) -> shap.Explanation:
        """
        Explain a single ``DocSample`` using SHAP.

        Parameters
        ----------
        sample : DocSample
            The sample to explain.
        align_boxes : bool, optional
            If True, masked tokens collapse their bounding boxes to the
            full page.  This is useful when you want to ignore the
            positional information of masked tokens.  Defaults to False.
        num_samples : int, optional
            The maximum number of model evaluations.  Higher values
            yield more accurate SHAP estimates but increase runtime.

        Returns
        -------
        shap.Explanation
            An object containing SHAP values, base values and data for
            the explained sample.
        """
        # build the prediction function for the given sample
        fn = self._make_predict_fn(sample, align_boxes)
        # build a Text masker; collapse_mask_token='auto' allows SHAP to
        # decide when to collapse consecutive masked tokens
        masker = shap.maskers.Text(
            tokenizer=self.tokenizer,
            mask_token=self.mask_token,
            collapse_mask_token='auto',
        )
        # build the explainer; we do not specify a link because the
        # prediction function already returns log‑odds
        explainer = shap.Explainer(
            fn,
            masker=masker,
            algorithm=self.algorithm,
            output_names=self.class_names,
        )
        # construct a sentinel‑delimited document for SHAP; join on space
        doc = " ".join(sample.words)
        # compute and return the explanation for a single document
        explanation = explainer([doc], max_evals=num_samples)[0]
        return explanation


class SHAPLayoutExplainer(SHAPTextExplainer):
    """
    SHAP explainer for the layout modality.

    This explainer extends ``SHAPTextExplainer`` by masking both words
    and their associated bounding boxes.  When a token is masked, its
    bounding box is replaced by the full page, so the model cannot use
    positional information to infer the missing content.
    """

    def _make_predict_fn(self, sample: DocSample, align_boxes: bool | None = None):
        # ignore align_boxes; boxes are always collapsed for masked tokens
        def fn(z_bin_mat: np.ndarray) -> np.ndarray:
            perturbed: list[DocSample] = []
            w, h = sample.image.size
            for z in z_bin_mat:
                # keep words or replace with mask token (same as text SHAP)
                words = [wrd if keep else self.mask_token for wrd, keep in zip(sample.words, z)]
                # replace bounding box with full page for masked tokens
                boxes = [b if keep else [0, 0, w, h] for b, keep in zip(sample.bboxes, z)]
                perturbed.append(
                    DocSample(
                        image=sample.image,
                        words=words,
                        bboxes=boxes,
                        ner_tags=sample.ner_tags,
                        label=sample.label,
                    )
                )
            return self._batched_predict(perturbed)
        return fn

    def explain(self, sample: DocSample, num_samples: int = 2000) -> shap.Explanation:
        """Explain a single sample for the layout modality."""
        fn = self._make_predict_fn(sample)
        masker = shap.maskers.Text(tokenizer=sentinel_tokenizer, mask_token=self.mask_token)
        explainer = shap.Explainer(fn, masker=masker, algorithm=self.algorithm, output_names=self.class_names)
        doc = DELIMITER.join(sample.words)
        return explainer([doc], max_evals=num_samples)[0]


class SHAPVisionExplainer(BaseShapExplainer):
    """
    SHAP explainer for the vision modality.

    This explainer masks out patches of an image by inpainting them and
    then feeds the masked images through the model.  It returns SHAP
    values over image pixels (or regions) representing the influence
    each patch has on the model's predictions.
    """

    def __init__(self, model, encode_fn, *, label: int | None = None, mask_value: str = "inpaint_telea",
                 batch_size: int = 32, algorithm: str = "partition", device: str | None = None):
        super().__init__(model, encode_fn, algorithm=algorithm, device=device)
        self.mask_value = mask_value
        self.batch_size = batch_size
        self.label = label  # restrict probability vector if desired

    def _batched_predict(self, samples: list[DocSample]) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i: i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, template: DocSample):
        """
        Return a function ``f(images_np) → probs``.

        *images_np* is a ``(N, H, W, C)`` array emitted by SHAP.  We wrap
        each array into a ``DocSample``, keeping the *words* and
        *bboxes* from *template* unchanged.
        """

        def fn(imgs: np.ndarray) -> np.ndarray:
            perturbed: list[DocSample] = [
                DocSample(
                    image=Image.fromarray(img.astype(np.uint8)),
                    words=template.words,
                    bboxes=template.bboxes,
                    ner_tags=template.ner_tags,
                    label=template.label,
                )
                for img in imgs
            ]
            probs = self._batched_predict(perturbed)
            # optionally select a single class
            if self.label is not None:
                probs = probs[:, self.label][:, None]
            return probs

        return fn

    def explain(self, sample: DocSample, *, num_samples: int = 8000, max_evals: int | None = None, max_batch: int = 64) -> shap.Explanation:
        """
        Explain a ``DocSample`` using image masking.

        Parameters
        ----------
        sample : DocSample
            The sample whose image we want to explain.
        num_samples : int, optional
            The default number of samples used by SHAP.  If ``max_evals`` is
            provided, it overrides this value.
        max_evals : int or None, optional
            Maximum number of model evaluations.  If ``None``, defaults to
            ``num_samples``.
        max_batch : int, optional
            Maximum number of perturbed images per batch when calling
            the model.

        Returns
        -------
        shap.Explanation
            SHAP values over image pixels/regions.
        """
        # extract image as numpy array; ensure 3D shape
        img_np = np.asarray(sample.image)
        if img_np.ndim == 2:  # grayscale → (H, W, 1)
            img_np = img_np[..., None]
        # build image masker
        masker = shap.maskers.Image(self.mask_value, shape=img_np.shape)
        explainer = shap.Explainer(
            self._make_predict_fn(sample),
            masker=masker,
            algorithm=self.algorithm,
            output_names=self.class_names,
        )
        max_evals = max_evals or num_samples
        explanation = explainer([img_np], max_evals=max_evals, batch_size=max_batch)[0]
        return explanation