import torch
import numpy as np
import shap
from vrdu_utils.module_types import DocSample
from PIL import Image
from functools import wraps
from shap.maskers._masker import Masker
import nltk
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
    SHAP explainer for text modality, mirroring LimeTextExplainer structure.
    Uses shap.Explainer with Text masker and a custom predict_fn that
    applies encode_fn to DocSample perturbations.
    """

    def __init__(
        self,
        model,
        encode_fn,
        tokenizer,
        mask_token: str = "|~|",
        device: str = None,
        batch_size: int = 16,
        algorithm: str = 'partition'
    ):
        super().__init__(model, encode_fn, algorithm, device)
        self.mask_token = mask_token
        self.batch_size = batch_size
        self.algorithm = algorithm
        if isinstance(tokenizer, LayoutLMv3TokenizerFast):
           self.tokenizer = make_layoutlmv3_tokenizer_wrapper(tokenizer)
        else:
            self.tokenizer = tokenizer
    def _batched_predict(self, samples: list[DocSample]) -> np.ndarray:
        """
        Batch predictions over DocSample list to avoid OOM.
        """
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i : i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, sample: DocSample, align_boxes: bool = False):
        W, H = sample.image.size
        dummy_box = [0, 0, W, H]

        def fn(perturbed_texts: list[str]):
            ds_batch = []
            for sent in perturbed_texts:
                words = sent.split()  # SHAP separated tokens with spaces
                boxes = []

                for i, w in enumerate(words):
                    if i < len(sample.bboxes):
                        # keep original box or collapse it, depending on align_boxes
                        if align_boxes and w == self.mask_token:
                            boxes.append(dummy_box)
                        else:
                            boxes.append(sample.bboxes[i])
                    else:
                        # SHAP produced a word we never had – give it a dummy box
                        boxes.append(dummy_box)

                ds_batch.append(
                    DocSample(image=sample.image,
                              words=words,
                              bboxes=boxes,
                              ner_tags=sample.ner_tags,
                              label=sample.label)
                )
            return self._batched_predict(ds_batch)

        return fn

    def explain(
        self,
        sample: DocSample,
        outputs,
        align_boxes: bool = False,
        num_samples: int = 2000,

    ):
        """
        Explain one DocSample via shap.Explainer(Text masker).

        sample      : the DocSample to explain
        align_boxes : if True, masked tokens zero out their boxes
        num_samples : maximum evals (max_evals)

        Returns shap.Explanation for the sample.
        """
        fn = self._make_predict_fn(sample, align_boxes)
        # background: all-masked
        n_tokens = len(sample.words)
        background = np.random.randint(0, 2, size=(200, n_tokens)),
        masker = shap.maskers.Text(
            tokenizer=self.tokenizer,
            mask_token=self.mask_token,
            collapse_mask_token='auto'
        )
        print(self.algorithm)
        print("check")
        explainer = shap.Explainer(
            fn,
            masker=masker,
            algorithm=self.algorithm,
            link=shap.links.identity,
            outputs=outputs,

        )
        # build sentinel-delimited doc for SHAP
        doc = " ".join(sample.words)
        # compute and return first (and only) explanation
        return explainer([doc], max_evals=num_samples)
class BBoxMasker(Masker):
    """
    Custom masker that perturbs only bounding boxes (not words).
    Used for layout-only SHAP with 'permutation' algorithm.
    """

    def __init__(self, reference_words, mask_box=[0, 0, 0, 0]):
        super().__init__()
        self.words = reference_words
        self.mask_box = mask_box
        self._input_format = "list"
        self._output_format = "list"
        self._input_names = ["bboxes"]

    def __call__(self, masks: np.ndarray, inputs, **kwargs):
        """
        masks: [num_perturbations, num_tokens] binary matrix
        inputs: original bboxes (length = num_tokens)
        """
        out = []
        for mask in masks:
            perturbed_boxes = [box if keep else self.mask_box for box, keep in zip(inputs, mask)]
            out.append((self.words, perturbed_boxes))  # tuple with words fixed
        return out

    def invariants(self, inputs):
        return [False] * len(inputs)  # all boxes are perturbable

    def shape(self, inputs):
        return len(inputs)

    def data(self, inputs):
        return inputs

    def summarize(self, inputs):
        return self.words  # for token-wise display


class SHAPLayoutExplainer(BaseShapExplainer):
    """
    Explainer for layout-only SHAP using Monte Carlo sampling.
    Only bounding boxes are perturbed. Words are kept constant.
    """

    def __init__(self, model, encode_fn, device=None, mask_box=[0, 0, 0, 0]):
        super().__init__(model, encode_fn, algorithm="permutation", device=device)
        self.mask_box = mask_box
        self.reference_sample = None  # must be set before explanation
        self.explainer = None  # initialized later

    def _predict_bbox_only(self, masked_inputs):
        """
        Construct full DocSamples using fixed words and perturbed boxes.
        """
        assert self.reference_sample is not None
        hydrated = [
            DocSample(
                image=self.reference_sample.image,
                words=self.reference_sample.words,        # fixed
                bboxes=boxes,                             # perturbed
                ner_tags=self.reference_sample.ner_tags,
                label=self.reference_sample.label,
            )
            for (_, boxes) in masked_inputs
        ]
        return self._predict(hydrated)

    def explain(self, sample: DocSample, num_samples: int = 512):
        """
        Run SHAP using only layout perturbation (words fixed).
        """
        self.reference_sample = sample
        masker = BBoxMasker(reference_words=sample.words, mask_box=self.mask_box)

        self.explainer = shap.Explainer(
            self._predict_bbox_only,
            masker,
            algorithm="permutation",
            output_names=self.class_names,
            link=shap.links.logit,
        )
        return self.explainer(sample.bboxes, max_evals=num_samples)

class SHAPVisionExplainer(BaseShapExplainer):
    def __init__(self,model,encode_fn,*,label = None,mask_value = "inpaint_telea",batch_size = 32,algorithm = "partition",device = None,):
        super().__init__(model, encode_fn, algorithm=algorithm, device=device)
        self.mask_value = mask_value
        self.batch_size = batch_size
        self.label = label  # restrict probability vector if desired

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    def _batched_predict(self, samples):
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i : i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, template: DocSample):
        """Return f(images_np) → probs.

        *images_np* is a `(N, H, W, C)` array emitted by SHAP.  We wrap each array
        into a `DocSample`, keeping the *words* and *bboxes* from *template*
        unchanged.
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

    def explain(self,sample: DocSample,*,num_samples = 8000,max_evals = None, max_batch = 64):
        img_np = np.asarray(sample.image)
        if img_np.ndim == 2:  # greyscale → (H, W, 1)
            img_np = img_np[..., None]

        masker = shap.maskers.Image(self.mask_value, shape=img_np.shape)
        explainer = shap.Explainer(
            self._make_predict_fn(sample),
            masker=masker,
            algorithm=self.algorithm,
            output_names=self.class_names,
        )

        # SHAP expects a *batch* → wrap in list
        max_evals = max_evals or num_samples
        explanation = explainer([img_np], max_evals=max_evals, batch_size=max_batch)[0]
        return explanation
