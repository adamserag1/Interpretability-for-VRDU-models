import torch
import numpy as np
import shap
from vrdu_utils.module_types import DocSample

DELIMITER = "|~|"

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
        logits = self.model(**self._encode(samples)).logits
        if temp:
            scaled_logits = logits
            # scaled_logits = logits / temp
        return torch.softmax(scaled_logits, dim=-1).cpu().numpy()

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
        mask_token: str = "[UNK]",
        device: str = None,
        batch_size: int = 16,
    ):
        super().__init__(model, encode_fn, device)
        self.mask_token = mask_token
        self.batch_size = batch_size

    def _batched_predict(self, samples: list[DocSample]) -> np.ndarray:
        """
        Batch predictions over DocSample list to avoid OOM.
        """
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i : i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, sample: DocSample, align_boxes: bool):
        """
        Returns a function f(z_matrix) -> np.ndarray, where z_matrix is
        array of shape (n_perturb, n_tokens) of 0/1 indicating kept tokens.
        """
        def fn(z_bin_mat: np.ndarray) -> np.ndarray:
            perturbed = []
            w, h = sample.image.size
            for z in z_bin_mat:
                # mask tokens
                words = [wrd if keep else self.mask_token
                         for wrd, keep in zip(sample.words, z)]
                # optionally zero out boxes
                if align_boxes:
                    boxes = [b if keep else [0, 0, w, h]
                             for b, keep in zip(sample.bboxes, z)]
                else:
                    boxes = sample.bboxes
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

    def explain(
        self,
        sample: DocSample,
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
        background = np.zeros((1, len(sample.words)))
        masker = shap.maskers.Text(
            tokenizer=sentinel_tokenizer,
            mask_token=self.mask_token,
        )
        explainer = shap.Explainer(
            fn,
            masker=masker,
            algorithm=self.algorithm,
            outputs=[sample.label],
            output_names=self.class_names,
        )
        # build sentinel-delimited doc for SHAP
        doc = DELIMITER.join(sample.words)
        # compute and return first (and only) explanation
        return explainer([doc], max_evals=num_samples)[0]