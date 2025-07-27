import torch
import numpy as np
import shap
from vrdu_utils.module_types import DocSample
from PIL import Image
import nltk
from transformers import LayoutLMv3TokenizerFast

def make_layoutlmv3_tokenizer_wrapper(base_tokenizer):
    def wrapped(texts, **kwargs):
        # Handle SHAP's special call to tokenizer("")
        if isinstance(texts, str) and texts.strip() == "":
            return {"input_ids": [base_tokenizer.cls_token_id, base_tokenizer.sep_token_id]}

        # Normal SHAP usage — a real sentence like "the dog barked"
        if isinstance(texts, str):
            tokens = texts.strip().split()
            dummy_boxes = [[0, 0, 0, 0]] * len(tokens)
            return base_tokenizer(tokens, boxes=dummy_boxes, **kwargs)

        elif isinstance(texts, list) and isinstance(texts[0], str):
            token_lists = [t.strip().split() for t in texts]
            dummy_boxes = [[[0, 0, 0, 0]] * len(toks) for toks in token_lists]
            return base_tokenizer(token_lists, boxes=dummy_boxes, **kwargs)

        else:
            raise ValueError("SHAP-wrapper: input must be str or list of str")
    return wrapped

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
        tokenizer,
        mask_token: str = "[UNK]",
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

    def _make_predict_fn(self, sample: DocSample, align_boxes: bool):
        """
        Returns a function f(z_matrix) -> np.ndarray, where z_matrix is
        array of shape (n_perturb, n_tokens) of 0/1 indicating kept tokens.
        """
        def fn(z_bin_mat: np.ndarray) -> np.ndarray:
            perturbed = []
            w, h = sample.image.size
            print(f'SHAP{z_bin_mat}')
            for z in z_bin_mat:
                words = [wrd if keep else self.mask_token for wrd, keep in zip(sample.words, z)]
                print(len(words))
                print(len(sample.boxes))
                print(words)
                # optionally zero out boxes
                if align_boxes:
                    boxes = [b if keep else [0, 0, w, h] for b, keep in zip(sample.bboxes, z)]
                else:
                    boxes = sample.bboxes
                perturbed.append(
                    DocSample(
                        image=sample.image,
                        words=words,
                        bboxes=boxes,#[:(len(words))],
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
        print(sample.words)
        masker = shap.maskers.Text(
            tokenizer=self.tokenizer,
            mask_token=self.mask_token,
            collapse_mask_token=False
        )
        print(self.algorithm)

        explainer = shap.Explainer(
            fn,
            masker=masker,
            algorithm=self.algorithm,
            # outputs=[sample.label],
            output_names=self.class_names,
        )
        # build sentinel-delimited doc for SHAP
        doc = " ".join(sample.words)
        # compute and return first (and only) explanation
        print([doc])
        return explainer([doc], max_evals=num_samples)[0]

class SHAPLayoutExplainer(SHAPTextExplainer):
    def _make_predict_fn(self, sample: DocSample, align_boxes: bool = None):
        def fn(z_bin_mat: np.ndarray) -> np.ndarray:
            perturbed = []
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

    def explain(self, sample: DocSample, num_samples: int = 2000):
        fn = self._make_predict_fn(sample)
        masker = shap.maskers.Text(tokenizer=sentinel_tokenizer, mask_token=self.mask_token)
        explainer = shap.Explainer(fn, masker=masker, algorithm=self.algorithm, output_names=self.class_names)
        doc = DELIMITER.join(sample.words)
        return explainer([doc], max_evals=num_samples)[0]



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
