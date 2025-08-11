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


# work around
def make_layoutlmv3_tokenizer_wrapper(tkn,
                                      dummy_box=(0, 0, 0, 0)):
    def _to_words(x):
        return list(x) if isinstance(x, (list, tuple)) else x.split()

    @wraps(tkn)
    def wrapped(texts, **kwargs):
        if isinstance(texts, str):
            words = _to_words(texts)
            if "boxes" not in kwargs:
                kwargs["boxes"] = [dummy_box] * len(words)
            return tkn(words,
                       **kwargs)


        batch_words = [_to_words(s) for s in texts]
        if "boxes" not in kwargs:
            kwargs["boxes"] = [
                [dummy_box] * len(seq) for seq in batch_words
            ]
        return tkn(batch_words,
                   **kwargs)

    return wrapped


class BaseShapExplainer:
    def __init__(self, model, encode_fn, algorithm = 'permutation', device=None):
        self.model = model.eval()
        self.encode_fn = encode_fn
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_names = [v for k, v in sorted(model.config.id2label.items())] if hasattr(model.config,
                                                                                            "id2label") else []
        self.algorithm = algorithm

        self.explainer: shap.Explainer = None

    def _encode(self, samples):
        return self.encode_fn(samples, self.device)

    @torch.no_grad()
    def _predict(self, samples, temp=1.0):
        # print(self.model(**self._encode(samples))) # Debugging
        try:
            # For LLMV3
            logits = self.model(**self._encode(samples)).logits
        except:
            # For BROS DocClass
            logits_loss_dict = self.model(**self._encode(samples))
            logits = logits_loss_dict['logits']
            # print(logits)
        if temp: # attempted with temperature scaling for logits
            scaled_logits = logits
            # scaled_logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        eps = 1e-16
        log_odds = torch.log(probs / (1.0 - probs * eps))
        # LOG ODDS AS NOTED BY SHAP (better for classification)
        return log_odds.cpu().numpy()

    def explain(self, data, **shap_kwargs):
        if self.explainer is None:
            raise ValueError("Explainer not initialised. Did you call super().__init__ and set up the masker?")
        return self.explainer(data, **shap_kwargs)

    def explain_batch(self, samples, **kwargs):
        return [self.explain(s, **kwargs) for s in samples]

class SHAPTextExplainer(BaseShapExplainer):

    def __init__(self,
                 model,
                 encode_fn,
                 tokenizer,
                 *,
                 mask_token = "|~|",
                 batch_size= 16,
                 device = None):
        super().__init__(model, encode_fn, algorithm="permutation", device=device)
        self.batch_size = batch_size
        self.mask_token = mask_token
        if isinstance(tokenizer, LayoutLMv3TokenizerFast):
            self.tokenizer = make_layoutlmv3_tokenizer_wrapper(tokenizer)
        else:
            self.tokenizer = tokenizer

    def _batched_predict(self, samples):
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i: i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, sample: DocSample, *, align_boxes: bool):

        w_page, h_page = sample.image.size
        full_box = [0, 0, w_page, h_page]

        def predict(z_mat):
            perturbed = []
            for z in z_mat.astype(int):
                words = [w if keep else self.mask_token
                         for w, keep in zip(sample.words, z)]
                # boxes = (template.bboxes if not align_boxes
                #          else [b if keep else full_box
                #                for b, keep in zip(template.bboxes, z)])
                perturbed.append(
                    DocSample(sample.image, words, sample.bboxes,
                              ner_tags=sample.ner_tags,
                              label=sample.label)
                )
            return self._batched_predict(perturbed)

        return predict


    def explain(self,
                sample: DocSample,
                *,
                nsamples= 2_000,
                align_boxes= False,
                random_state = None):

        n_tokens = len(sample.words)
        data_row = np.ones((1, n_tokens), dtype=int)              # batch-size 1
        baseline = np.zeros((1, n_tokens), dtype=int)             # all masked
        masker = shap.maskers.Independent(baseline)

        self.explainer = shap.Explainer(
            self._make_predict_fn(sample, align_boxes=align_boxes),
            masker,
            algorithm="permutation",
            output_names=self.class_names,
            link=shap.links.identity,   # return log-odds
            seed=random_state,
        )

        return self.explainer(data_row, max_evals=nsamples)

class SHAPLayoutExplainer(BaseShapExplainer):
    def __init__(self, model, encode_fn, *, batch_size= 16,
                 mask_full_page= True, device = None):
        super().__init__(model, encode_fn, algorithm="permutation", device=device)
        self.batch_size = batch_size
        self.mask_full_page = mask_full_page

    def _batched_predict(self, samples):
        out = []
        for i in range(0, len(samples), self.batch_size):
            out.append(self._predict(samples[i: i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, template: DocSample):
        w_page, h_page = template.image.size
        full_box = [0, 0, w_page, h_page]

        def predict(z_bin_mat: np.ndarray):
            perturbed = []
            for z_row in z_bin_mat.astype(int):
                boxes = [b if keep else (full_box if self.mask_full_page else [0, 0, 0, 0])
                         for b, keep in zip(template.bboxes, z_row)]
                perturbed.append(
                    DocSample(
                        image=template.image,
                        words=template.words,
                        bboxes=boxes,  # perturbed layout
                        ner_tags=template.ner_tags,
                        label=template.label,
                    )
                )
            return self._batched_predict(perturbed)

        return predict

    def explain(self, sample: DocSample, *,
                nsamples = 2_000, random_state = None):
        n_tokens = len(sample.bboxes)

        data_row = np.ones((1,n_tokens), dtype=int)
        baseline = np.zeros((1, n_tokens), dtype=int)

        masker = shap.maskers.Independent(baseline)

        self.explainer = shap.Explainer(
            self._make_predict_fn(sample),
            masker,
            algorithm="permutation",
            output_names=self.class_names,
            link=shap.links.identity,
            seed=random_state,
        )

        return self.explainer(data_row, max_evals=nsamples)

class SHAPVisionExplainer(BaseShapExplainer):
    def __init__(self,
                 model,
                 encode_fn,
                 class_idx,
                 *,
                 mask_value= "inpaint_ns",
                 batch_size = 32,
                 device = None,
                 ):
        super().__init__(model, encode_fn, algorithm="permutation", device=device)
        self.mask_value = mask_value
        self.batch_size = batch_size
        self.class_idx = class_idx

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
        class_idx = self.class_idx
        def predict(img_batch):
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
                nsamples= 8_000,
                random_state = None,
                max_batch= 64):

        img_np = np.asarray(sample.image.convert('RGB'))
        if img_np.ndim == 2:
            img_np = img_np[..., None]

        assert isinstance(self.class_names[self.class_idx], str)
        masker = shap.maskers.Image(self.mask_value, shape=img_np.shape)
        self.explainer = shap.Explainer(
            self._make_predict_fn(sample),
            masker,
            algorithm="partition",
            link=shap.links.identity,
            seed=random_state,
            batch_size=max_batch,
        )

        return self.explainer(
            np.expand_dims(img_np, 0),
            max_evals=nsamples,
            batch_size=max_batch,
        )[0]


class NerAdapter:
    def __init__(self, *args, target_token_fn, target_labels= None, **kwargs):
        self.target_token_fn = target_token_fn
        self.target_labels = target_labels
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _predict(self, samples):
        enc, _ = self._encode(samples)
        out = self.model(**enc).logits  # (batch, seq, n_labels)
        probs = torch.softmax(out, dim=-1).cpu().numpy()

        tok_idx = self.target_token_fn(enc)
        tok_probs = probs[:, tok_idx, :]

        return tok_probs
class SHAPTextNer(NerAdapter, SHAPTextExplainer):
    pass

class SHAPLayoutNer(NerAdapter, SHAPLayoutExplainer):
    pass

class SHAPVisionNer(NerAdapter, SHAPVisionExplainer):
    pass
