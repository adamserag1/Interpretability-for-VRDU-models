"""
SHAP explainers tailored for visually rich document understanding (VRDU).

This module provides three explainers built on top of the SHAP library:

* **SHAPTextExplainer** – explains word‑level contributions in documents.  It
  uses a custom tokenizer that treats each word in a `DocSample` as a token
  separated by a sentinel delimiter.  Unlike the original implementation, this
  tokenizer preserves empty tokens and disables mask–collapse behaviour in
  `shap.maskers.Text` so that the number of mask positions exactly matches
  the number of words.  Predictions are made by masking tokens in the
  underlying `DocSample` and forwarding them through the model.

* **SHAPLayoutExplainer** – explains layout (bounding‑box) importance.  It
  shares the same masking scheme as the text explainer but always zeros out
  the bounding boxes of masked tokens.  This allows attributions to reflect
  spatial importance rather than lexical content.

* **SHAPVisionExplainer** – explains pixel‑level contributions in images.  It
  wraps page images emitted by a `DocSample` in an image masker and passes
  them through the model.  Words and bounding boxes remain fixed across
  perturbed images.

For all explainers, the returned `shap.Explanation` object contains
per‑feature SHAP values along with the original (sentinel‑delimited) data.

Note
----
SHAP’s text masker collapses consecutive masked tokens into a single mask
token by default (`collapse_mask_token="auto"`), which reduces the number
of features in the explanation【642620058882920†L83-L87】.  Here we set
`collapse_mask_token=False` to preserve one feature per word.  The custom
tokenizer also returns a full `offset_mapping` so that SHAP knows how to
align tokens with the original string【642620058882920†L69-L76】.
"""

import torch
import numpy as np
import shap
from vrdu_utils.module_types import DocSample
from PIL import Image

# Sentinel delimiter used to join and split tokens in the document.  A
# sequence of words is joined by this delimiter before being passed to SHAP,
# and split back into tokens when needed.  It should not appear within
# individual tokens.
DELIMITER = "|~|"

def safe_split(doc_str: str) -> list[str]:
    """Split a sentinel‑delimited document string into tokens.

    This function splits on every occurrence of :data:`DELIMITER` and
    **preserves empty tokens**, which ensures that the number of tokens
    produced exactly matches the number of words originally joined.  It is
    complementary to :func:`sentinel_tokenizer`.

    Parameters
    ----------
    doc_str : str
        The sentinel‑delimited string to split.

    Returns
    -------
    list[str]
        A list of tokens, where empty strings correspond to positions in the
        original word list that were empty.
    """
    return doc_str.split(DELIMITER)


def sentinel_tokenizer(s: str, return_offsets_mapping: bool = True, **kw) -> dict:
    """Custom tokenizer for SHAP's Text masker.

    The tokenizer splits the input string on the sentinel delimiter and returns
    ``input_ids`` that enumerate the positions of the resulting tokens.  Unlike
    the earlier implementation, no tokens are filtered out.  When
    ``return_offsets_mapping`` is ``True`` the function also returns
    ``offset_mapping``—a list of ``(start, end)`` character positions—that
    allows SHAP to reconstruct the original substrings.

    Parameters
    ----------
    s : str
        The sentinel‑delimited string to tokenize.
    return_offsets_mapping : bool, optional
        Whether to include offset mapping information.  SHAP uses this
        mapping to build its maskers.【642620058882920†L69-L76】  Defaults to ``True``.

    Returns
    -------
    dict
        A dictionary containing ``input_ids`` and optionally ``offset_mapping``.
    """
    # Split the string on every delimiter, preserving empty tokens
    tokens = s.split(DELIMITER)
    ids = list(range(len(tokens)))
    if return_offsets_mapping:
        offsets = []
        pos = 0
        for t in tokens:
            # ``pos`` is the starting character index of the current token in
            # the concatenated string.  The end index is ``pos + len(t)``.
            offsets.append((pos, pos + len(t)))
            # Advance the position by the token length plus the delimiter
            pos += len(t) + len(DELIMITER)
        return {"input_ids": ids, "offset_mapping": offsets}
    return {"input_ids": ids}


class BaseShapExplainer:
    """Common functionality for SHAP explainers.

    This base class wraps a model and encoding function.  It exposes a
    ``_predict`` method that forwards a list of :class:`DocSample` objects
    through the model and returns probability vectors.  Child classes
    implement modality‑specific masking schemes.
    """

    def __init__(self, model, encode_fn, algorithm: str = "partition", device: str | None = None):
        self.model = model.eval()
        self.encode_fn = encode_fn
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        # Extract class names from the model config if available
        self.class_names = [v for k, v in sorted(model.config.id2label.items())] if hasattr(model.config, "id2label") else []
        self.algorithm = algorithm

    def _encode(self, samples: list[DocSample]):
        """Apply the provided encoding function on a list of samples."""
        return self.encode_fn(samples, self.device)

    @torch.no_grad()
    def _predict(self, samples: list[DocSample], temp: float = 1.0) -> np.ndarray:
        """Forward a batch of samples through the model and return probabilities."""
        try:
            logits = self.model(**self._encode(samples)).logits
        except Exception:
            logits_loss_dict = self.model(**self._encode(samples))
            logits = logits_loss_dict["logits"]
        scaled_logits = logits if temp else logits  # reserved for potential temperature scaling
        return torch.softmax(scaled_logits, dim=-1).cpu().numpy()

    def explain(self, data, **shap_kwargs):
        """Compute SHAP values for the given data using the configured explainer.

        Child classes should set ``self.explainer`` before invoking this.
        """
        if getattr(self, "explainer", None) is None:
            raise ValueError("Explainer not initialised. Did you call the appropriate explain method?")
        return self.explainer(data, **shap_kwargs)

    def explain_batch(self, samples: list[DocSample], **kwargs):
        """Explain a list of samples by invoking ``explain`` on each."""
        return [self.explain(s, **kwargs) for s in samples]


class SHAPTextExplainer(BaseShapExplainer):
    """SHAP explainer for the text modality.

    This explainer perturbs individual words in a document by replacing masked
    tokens with a special mask token.  When ``align_boxes`` is ``True``, it also
    zeros out the corresponding bounding boxes; otherwise bounding boxes remain
    intact.  The underlying model function accepts a list of binary masks
    (one per perturbation) and returns prediction probabilities.
    """

    def __init__(
        self,
        model,
        encode_fn,
        mask_token: str = "[UNK]",
        device: str | None = None,
        batch_size: int = 16,
        algorithm: str = "partition",
    ):
        super().__init__(model, encode_fn, algorithm, device)
        self.mask_token = mask_token
        self.batch_size = batch_size

    def _batched_predict(self, samples: list[DocSample]) -> np.ndarray:
        """Batch predictions over ``DocSample`` lists to avoid GPU OOM errors."""
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i : i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, sample: DocSample, align_boxes: bool) -> callable:
        """Return a function mapping binary masks to model predictions.

        The returned function takes a 2‑D binary array ``z_bin_mat`` of shape
        ``(n_perturb, n_tokens)``.  Each row corresponds to a perturbation and
        indicates whether a token is kept (1) or masked (0).  For each
        perturbation the appropriate ``DocSample`` is constructed and passed
        through the model.  Boxes are zeroed if ``align_boxes`` is ``True``.
        """
        def fn(z_bin_mat: np.ndarray) -> np.ndarray:
            perturbed: list[DocSample] = []
            width, height = sample.image.size
            for z in z_bin_mat:
                # Build the list of words and boxes for this perturbation
                words = [wrd if keep else self.mask_token for wrd, keep in zip(sample.words, z)]
                if align_boxes:
                    boxes = [b if keep else [0, 0, width, height] for b, keep in zip(sample.bboxes, z)]
                else:
                    boxes = list(sample.bboxes)
                # Ensure the number of boxes matches the number of words
                if len(boxes) > len(words):
                    boxes = boxes[: len(words)]
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

    def explain(self, sample: DocSample, align_boxes: bool = False, num_samples: int = 2000):
        """Explain a single document sample using SHAP.

        Parameters
        ----------
        sample : DocSample
            The document to explain.
        align_boxes : bool, optional
            If ``True``, bounding boxes corresponding to masked words are
            replaced with the full page box.  Defaults to ``False``.
        num_samples : int, optional
            Maximum number of evaluations (``max_evals``) used by SHAP.

        Returns
        -------
        shap.Explanation
            SHAP explanation object with per‑token attributions.
        """
        # Build the predict function that accepts binary masks
        fn = self._make_predict_fn(sample, align_boxes)
        # Build a Text masker that preserves one feature per word
        masker = shap.maskers.Text(
            tokenizer=sentinel_tokenizer,
            mask_token=self.mask_token,
            collapse_mask_token=False,  # preserve a mask position for every word【642620058882920†L83-L87】
        )
        # Create the explainer
        explainer = shap.Explainer(
            fn,
            masker=masker,
            algorithm=self.algorithm,
            output_names=self.class_names,
        )
        # Prepare the document string for SHAP by joining words with the sentinel delimiter
        doc = DELIMITER.join(sample.words)
        # Compute and return the explanation; wrap the single document in a list
        return explainer([doc], max_evals=num_samples)[0]


class SHAPLayoutExplainer(SHAPTextExplainer):
    """SHAP explainer for layout (bounding‑box) importance.

    This subclass uses the same token masking logic as :class:`SHAPTextExplainer`
    but always zeros out the bounding box of a masked token, irrespective of
    the ``align_boxes`` argument.  Consequently, the attributions focus on
    spatial layout rather than token content.
    """

    def explain(self, sample: DocSample, num_samples: int = 2000):
        # Always align boxes for layout explanations
        fn = self._make_predict_fn(sample, align_boxes=True)
        masker = shap.maskers.Text(
            tokenizer=sentinel_tokenizer,
            mask_token=self.mask_token,
            collapse_mask_token=False,
        )
        explainer = shap.Explainer(
            fn,
            masker=masker,
            algorithm=self.algorithm,
            output_names=self.class_names,
        )
        doc = DELIMITER.join(sample.words)
        return explainer([doc], max_evals=num_samples)[0]


class SHAPVisionExplainer(BaseShapExplainer):
    """SHAP explainer for the vision modality.

    Images are perturbed by replacing pixels with a specified mask value and
    forwarded through the model.  Words and bounding boxes remain fixed
    throughout the perturbations.  See SHAP’s image masker documentation for
    details on the masking strategy.
    """

    def __init__(
        self,
        model,
        encode_fn,
        *,
        label: int | None = None,
        mask_value: str | tuple | int = "inpaint_telea",
        batch_size: int = 32,
        algorithm: str = "partition",
        device: str | None = None,
    ):
        super().__init__(model, encode_fn, algorithm=algorithm, device=device)
        self.mask_value = mask_value
        self.batch_size = batch_size
        self.label = label  # restrict probability vector if desired

    def _batched_predict(self, samples: list[DocSample]) -> np.ndarray:
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i : i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, template: DocSample):
        """Return a function that maps perturbed images to probabilities.

        The returned function accepts a numpy array of images (N, H, W, C) and
        wraps each image into a :class:`DocSample` using the words and
        bounding boxes from ``template``.  Predictions are computed via
        ``_batched_predict``.
        """
        def fn(imgs: np.ndarray) -> np.ndarray:
            perturbed = [
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
            if self.label is not None:
                probs = probs[:, self.label][:, None]
            return probs
        return fn

    def explain(
        self,
        sample: DocSample,
        *,
        num_samples: int = 8000,
        max_evals: int | None = None,
        max_batch: int = 64,
    ) -> shap.Explanation:
        """Explain a document's image using SHAP.

        Parameters
        ----------
        sample : DocSample
            The document page to explain.
        num_samples : int, optional
            Default number of evaluations if ``max_evals`` is not supplied.
        max_evals : int or None, optional
            Maximum number of evaluations for the explainer.  Overrides
            ``num_samples`` if provided.
        max_batch : int, optional
            Batch size used internally by SHAP when computing explanations.

        Returns
        -------
        shap.Explanation
            SHAP explanation object with per‑pixel attributions.
        """
        img_np = np.asarray(sample.image)
        # Convert grayscale images to shape (H, W, 1)
        if img_np.ndim == 2:
            img_np = img_np[..., None]
        masker = shap.maskers.Image(self.mask_value, shape=img_np.shape)
        explainer = shap.Explainer(
            self._make_predict_fn(sample),
            masker=masker,
            algorithm=self.algorithm,
            output_names=self.class_names,
        )
        # SHAP expects a list of images
        max_evals = max_evals or num_samples
        explanation = explainer([img_np], max_evals=max_evals, batch_size=max_batch)[0]
        return explanation