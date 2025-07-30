import torch
import numpy as np
import shap
from vrdu_utils.module_types import DocSample
from PIL import Image
from functools import wraps
from shap.maskers._masker import Masker
import nltk
from copy import deepcopy
from transformers import LayoutLMv3TokenizerFast


def make_layoutlmv3_tokenizer_wrapper(tkn: LayoutLMv3TokenizerFast,
                                      dummy_box=(0, 0, 0, 0)):
    """Return a callable that always feeds LayoutLM-v3 a
    (words, boxes, is_split_into_words=True) triple — even for the
    empty-string probe used by SHAP."""

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
            return tkn(words,
                       **kwargs)

        # ---------------- batch -------------------------
        batch_words = [_to_words(s) for s in texts]  # list[list[str]]
        if "boxes" not in kwargs:
            kwargs["boxes"] = [
                [dummy_box] * len(seq) for seq in batch_words
            ]
        return tkn(batch_words,
                   **kwargs)

    return wrapped

DELIMITER = " "

def safe_split(doc_str: str) -> list[str]:
    """Split a sentinel-delimited document string into tokens."""
    return doc_str.split(DELIMITER)


def sentinel_tokenizer(s: str, return_offsets_mapping=True, **kw):
    """Custom tokenizer for SHAP Text masker using sentinel delimiter."""
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
    def __init__(self, model, encode_fn, algorithm = 'partition', device=None):
        self.model = model.eval()
        self.encode_fn = encode_fn
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_names = [v for k, v in sorted(model.config.id2label.items())] if hasattr(model.config,
                                                                                            "id2label") else []
        self.algorithm = algorithm

        # the shap.Explainer instance
        self.explainer: shap.Explainer = None

    def _encode(self, samples):
        return self.encode_fn(samples, self.device)

    @torch.no_grad()
    def _predict(self, samples, temp=1.0):
        # print(self.model(**self._encode(samples))) # Debugging
        try:
            logits = self.model(**self._encode(samples)).logits
        except:
            logits_loss_dict = self.model(**self._encode(samples))
            logits = logits_loss_dict['logits']
            # print(logits)
        if temp:
            scaled_logits = logits
            # scaled_logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        eps = 1e-16
        log_odds = torch.log(probs / (1.0 - probs * eps))
        return log_odds.cpu().numpy()

    def explain(self, data, **shap_kwargs):
        if self.explainer is None:
            raise ValueError("Explainer not initialised. Did you call super().__init__ and set up the masker?")
        return self.explainer(data, **shap_kwargs)

    def explain_batch(self, samples, **kwargs):
        return [self.explain(s, **kwargs) for s in samples]

class SHAPTextExplainer(BaseShapExplainer):
    """
    Monte-Carlo (permutation) SHAP for *text* modality.
    Only tokens are perturbed; bounding boxes stay unchanged unless
    `align_boxes=True`, in which case masked tokens get a dummy page-size box.
    """

    def __init__(self,
                 model,
                 encode_fn,
                 tokenizer,
                 *,
                 mask_token: str = "|~|",
                 batch_size: int = 16,
                 device: str | None = None):
        super().__init__(model, encode_fn, algorithm="permutation", device=device)
        self.batch_size = batch_size
        self.mask_token = mask_token
        if isinstance(tokenizer, LayoutLMv3TokenizerFast):
            self.tokenizer = make_layoutlmv3_tokenizer_wrapper(tokenizer)
        else:
            self.tokenizer = tokenizer

    # ---------------------------------------------------------------- helpers
    def _batched_predict(self, samples):
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i: i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, template: DocSample, *, align_boxes: bool):
        """Return f(mask_matrix) → log-odds for permutation SHAP."""

        w_page, h_page = template.image.size
        full_box = [0, 0, w_page, h_page]

        def predict(z_mat: np.ndarray) -> np.ndarray:
            perturbed = []
            for z in z_mat.astype(int):
                words = [w if keep else self.mask_token
                         for w, keep in zip(template.words, z)]
                # boxes = (template.bboxes if not align_boxes
                #          else [b if keep else full_box
                #                for b, keep in zip(template.bboxes, z)])
                perturbed.append(
                    DocSample(template.image, words, template.bboxes,
                              ner_tags=template.ner_tags,
                              label=template.label)
                )
            return self._batched_predict(perturbed)

        return predict


    def explain(self,
                sample: DocSample,
                *,
                nsamples: int = 2_000,
                align_boxes: bool = False,
                random_state: int | None = None):
        """
        Monte-Carlo SHAP explanation of a single DocSample.

        Parameters
        ----------
        nsamples     : permutations to draw (trade-off speed vs. variance)
        align_boxes  : if True, masked tokens also zero-out their bounding box
        """

        n_tokens = len(sample.words)
        data_row = np.ones((1, n_tokens), dtype=int)              # batch-size 1
        baseline = np.zeros((1, n_tokens), dtype=int)             # all masked
        masker   = shap.maskers.Independent(baseline)             # 2-D required

        self.explainer = shap.Explainer(
            self._make_predict_fn(sample, align_boxes=align_boxes),
            masker,
            algorithm="permutation",
            output_names=self.class_names,
            link=shap.links.identity,   # we already return log-odds
            seed=random_state,
        )

        # hand the explainer the batch matrix directly (no list wrapper!)
        return self.explainer(data_row, max_evals=nsamples)

class SHAPLayoutExplainer(BaseShapExplainer):
    """
       Monte-Carlo (permutation) SHAP explainer for *layout* modality.
       Only bounding-box coordinates are perturbed; words are kept fixed.
       """

    def __init__(self, model, encode_fn, *, batch_size: int = 16,
                 mask_full_page: bool = True, device: str | None = None):
        super().__init__(model, encode_fn, algorithm="permutation", device=device)
        self.batch_size = batch_size
        self.mask_full_page = mask_full_page  # if False → zero-box instead

    # ---------------------------------------------------------------- helpers
    def _batched_predict(self, samples) -> np.ndarray:
        """
        Predict a list of DocSample objects in mini-batches to avoid OOM.
        """
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i: i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, template: DocSample):
        """
        Returns f(z_mask) -> log-odds, where
          z_mask  : (N, n_tokens) binary matrix produced by SHAP.
        Each 0 in z_mask means *mask this token's bounding box*.
        """
        w_page, h_page = template.image.size
        full_box = [0, 0, w_page, h_page]

        def predict(z_bin_mat: np.ndarray) -> np.ndarray:
            perturbed: list[DocSample] = []
            for z_row in z_bin_mat.astype(int):
                boxes = [b if keep else (full_box if self.mask_full_page else [0, 0, 0, 0])
                         for b, keep in zip(template.bboxes, z_row)]
                perturbed.append(
                    DocSample(
                        image=template.image,
                        words=template.words,  # unchanged
                        bboxes=boxes,  # perturbed layout
                        ner_tags=template.ner_tags,
                        label=template.label,
                    )
                )
            return self._batched_predict(perturbed)

        return predict

    # ---------------------------------------------------------------- public
    def explain(self, sample: DocSample, *,
                nsamples: int = 2_000, random_state: int | None = None):
        """
        Compute a Monte-Carlo SHAP explanation for a single DocSample.
        `nsamples`    – SHAP permutations to draw (higher ⇒ lower variance).
        """
        n_tokens = len(sample.bboxes)
        # 1 = keep, 0 = mask
        data_row = np.ones((1,n_tokens), dtype=int)
        baseline = np.zeros((1, n_tokens), dtype=int) # what the masker inserts

        masker = shap.maskers.Independent(baseline)

        self.explainer = shap.Explainer(
            self._make_predict_fn(sample),
            masker,
            algorithm="permutation",
            output_names=self.class_names,
            link=shap.links.identity,  # already returning log-odds
            seed=random_state,
        )

        # SHAP expects a *batch* → wrap data_row in list
        return self.explainer(data_row, max_evals=nsamples)

class SHAPVisionExplainer(BaseShapExplainer):
    """
       Monte-Carlo (permutation) SHAP explainer for *vision* modality.

       • Only the image is perturbed (words + bboxes stay fixed).
       • Uses `shap.maskers.Image` with inpainting (default `mask_value="inpaint_telea"`).
       • Works in log-odds space via BaseShapExplainer._predict.
       """

    def __init__(self,
                 model,
                 encode_fn,
                 class_idx,
                 *,
                 mask_value: str = "inpaint_ns",
                 batch_size: int = 32,
                 device: str | None = None,
                 ):
        super().__init__(model, encode_fn, algorithm="permutation", device=device)
        self.mask_value = mask_value
        self.batch_size = batch_size
        self.class_idx = class_idx

    # ---------------------------------------------------------------- helpers
    def _batched_predict(self, samples):
        return self._predict(samples)
        # if len(samples) <= self.batch_size:
        #     return self._predict(samples)  # ← return!
        # # slow path: rare fallback
        # out = []
        # for i in range(0, len(samples), self.batch_size):
        #     out.append(self._predict(samples[i: i + self.batch_size]))
        # if out.shape[1] == 1:  # ← binary or single-output case
        #     out = out.squeeze(-1)
        # return np.vstack(out)

    def _make_predict_fn(self, template: DocSample):
        words, bboxes = template.words, template.bboxes
        ner, label = template.ner_tags, template.label
        class_idx = self.class_idx  # single int

        def predict(img_batch: np.ndarray) -> np.ndarray:
            perturbed = [
                DocSample(Image.fromarray(arr.astype(np.uint8)),
                          words, bboxes, ner_tags=ner, label=label)
                for arr in img_batch
            ]
            # out = self._batched_predict(perturbed)  # (N, C)
            # out = out[:, class_idx]  # → (N,)
            out = self._batched_predict(perturbed)
            return out[:, self.class_idx]
            # return self._batched_predict(perturbed)
        return predict


    def explain(self,
                sample: DocSample,
                *,
                nsamples: int = 8_000,
                random_state: int | None = None,
                max_batch: int = 64):
        """
        Compute Monte-Carlo SHAP for a single `DocSample`.

        Parameters
        ----------
        nsamples : number of feature evaluations (permutation samples)
        """

        img_np = np.asarray(sample.image.convert('RGB'))
        if img_np.ndim == 2:  # greyscale → add channel
            img_np = img_np[..., None]

        assert isinstance(self.class_names[self.class_idx], str)
        masker = shap.maskers.Image(self.mask_value, shape=img_np.shape)
        self.explainer = shap.Explainer(
            self._make_predict_fn(sample),
            masker,
            # output_names = self.class_names[self.class_idx],
            algorithm="partition",
            link=shap.links.identity,  # _predict already returns log-odds
            seed=random_state,
            batch_size=max_batch,
        )

        # SHAP expects batch input → wrap in array of shape (1, H, W, C)
        return self.explainer(
            np.expand_dims(img_np, 0),
            max_evals=nsamples,
            batch_size=max_batch,
            # outputs = [self.class_idx]
        )[0]


class NerAdapter:
    def __init__(self, *args, target_token_fn, target_labels= None, **kwargs):
        self.target_token_fn = target_token_fn
        self.target_labels = target_labels
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _predict(self, samples):
        enc, _ = self._encode([samples], device)
        out = self.model(**enc)
        probs = torch.softmax(out, dim=-1).cpu.numpy()

        tok_idx = self.target_token_fn(enc)
        tok_probs = probs[:, tok_idx, :]

        if getattr(self, "target_labels", None) is None:
            chosen = tok_probs.max(dim=-1).values[:, None]
        else:
            chosen = tok_probs[: self.target_labels]
            if chosen.ndim == 1:
                chosen = chosen[:, None]
        if isinstance(chosen, torch.Tensor):
            return chosen.cpu().numpy()
        return chosen

class SHAPTextNer(NerAdapter, SHAPTextExplainer):
    pass

class SHAPLayoutNer(NerAdapter, SHAPLayoutExplainer):
    pass

class SHAPVisionNer(NerAdapter, SHAPVisionExplainer):
    pass
