"""
Applying Lime w.r.t VRDU modalities 
"""

import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer
from pygments.formatters import img
from tqdm import tqdm
from PIL import Image
from skimage.segmentation import slic
from lime.lime_image import LimeImageExplainer


from vrdu_utils.module_types import DocSample

"""
DOC-CLASS EXPLANINERS
"""

class BaseLimeExplainer:
    def __init__(self, model, encode_fn, device=None):
        self.model = model.eval()
        self.encode_fn = encode_fn
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_names = [v for k, v in sorted(model.config.id2label.items())] if hasattr(model.config, "id2label") else []
    
    def _encode(self, samples):
        return self.encode_fn(samples, self.device)

    @torch.no_grad
    def _predict(self, samples):
        logits = self.model(**self._encode(samples)).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def explain(self, sample, **kwargs):
        return NotImplementedError

    def explain_batch(self, samples, **kwargs):
        return [self.explain(s, **kwargs) for s in samples]


class LimeTextExplainer(BaseLimeExplainer):
    def __init__(self, model, encode_fn, mask_token = "[UNK]", device=None, *, kernel_width_factor=1.0, labels = None, batch_size = 16):
        super().__init__(model, encode_fn, device)
        self.mask_token = mask_token
        self.kernel_width_factor = kernel_width_factor
        self.batch_size = batch_size
        self.labels = labels

    def _batched_predict(self, samples):
        out = []
        itr = tqdm(range(0, len(samples), self.batch_size), desc = "[LIME] - Text")
        for i in itr:
            out.append(self._predict(samples[i:i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, sample: DocSample, align_boxes):
        def fn(z_bin_list):
            perturbed = []
            w, h = sample.image.size
            for z in z_bin_list:
                words = [w if z_i else self.mask_token for w, z_i in zip(sample.words, z)]
                if align_boxes:
                    boxes = [b if z_i else [0,0,w,h] for b, z_i in zip(sample.bboxes, z)]
                else:
                    boxes = sample.bboxes
                perturbed.append(DocSample(sample.image, words, boxes, ner_tags = sample.ner_tags, label=sample.label))
            print("MADE PREDICT")
            return self._batched_predict(perturbed)
        return fn

    # CHANGE NUMBER OF SMAPLES NUMBER OF FEATURES

    def explain(self, sample: DocSample, align_boxes = False, num_samples = 4000, num_features=30):
        n_tokens = len(sample.words)
        print("Begging EXPLAINER")
        explainer = LimeTabularExplainer(
            #training_data = np.vstack([np.ones(n_tokens), np.zeros(n_tokens)]),
            training_data=np.random.randint(0, 2, size=(200, n_tokens)),
            feature_names = sample.words,
            class_names = self.class_names,
            discretize_continuous = False,
            categorical_features = list(range(n_tokens)),
            mode = 'classification',
            kernel_width = np.sqrt(n_tokens) * self.kernel_width_factor
        )
        print("Begging EXPLAIN_INSTANCE")
        if labels:
            return explainer.explain_instance(
                data_row = np.ones(n_tokens),
                predict_fn=self._make_predict_fn(sample, align_boxes),
                num_samples = num_samples,
                num_features = num_features,
                labels = self.labels
            )
        else:
            return explainer.explain_instance(
                data_row = np.ones(n_tokens),
                predict_fn=self._make_predict_fn(sample, align_boxes),
                num_samples = num_samples,
                num_features = num_features,
            )

class LimeLayoutExplainer(LimeTextExplainer):
    """
    Perturbs **bounding boxes** only.
    Tokens (`words`) stay fixed, so attribution
    reflects *layout* importance, not lexical content.
    """
    # ---------- override only the perturb-function -------------------
    def _make_predict_fn(self, sample: DocSample, align_boxes=None):
        def fn(z_bin_mat):
            perturbed = []

            w, h = sample.image.size

            for z in z_bin_mat:
                boxes = [
                    b if keep else [0, 0, w, h]
                    for b, keep in zip(sample.bboxes, z)
                ]
                perturbed.append(
                    DocSample(
                        sample.image,                 # same page image
                        sample.words,                 # keep tokens
                        boxes,                        # modified boxes
                        label    = sample.label,
                        ner_tags = sample.ner_tags    # keep BIO labels
                    )
                )
            return self._batched_predict(perturbed)
        return fn

class LimeVisionExplainer(BaseLimeExplainer):
    def __init__(self, model, encode_fn, *, n_segments = 200, compactness = 20.0, sigma = 1.0, batch_size = 32, device = None):
        super().__init__(model, encode_fn, device)
        self.seg_kwargs = dict(
            n_segments = n_segments,
            compactness = compactness,
            sigma = sigma,
            start_label = 1
        )
        self.batch_size = batch_size

    def _batched_predict(self, samples):
        out = []
        it = range(0, len(samples), self.batch_size)
        from tqdm.auto import tqdm
        it = tqdm(it, desc="[LIME] - VISION", leave=False)
        for i in it:
            out.append(self._predict(samples[i:i + self.batch_size]))
        return np.vstack(out)

    def _make_predict_fn(self, sample: DocSample):
        def fn(img_list):
            perturbed = [
                DocSample(
                    Image.fromarray(arr),
                    sample.words,
                    sample.bboxes,
                    label = sample.label,
                    ner_tags = sample.ner_tags,
                )
                for arr in img_list
            ]
            return self._batched_predict(perturbed)

        return fn


    def explain(self, sample, *, num_samples = 8000, num_features = 30):
        explainer = LimeImageExplainer(random_state=0)
        img_np = np.array(sample.image)

        return explainer.explain_instance(
            img_np,
            classifier_fn = self._make_predict_fn(sample),
            segmentation_fn = lambda img: slic(img, **self.seg_kwargs),
            top_labels = 1,
            hide_color=(127, 127, 127),
            num_samples = num_samples,
            batch_size = self.batch_size
        )



"""
NER Explainers (adapters of ^^)
"""

class NerAdapter:
    """
    Mixin that adapts a document level Lime Explainer subclass for token-classification (NER FUNSD)
    """
    def __init__(self, *args, target_token_fn, target_labels=None, **kwargs):
        self.target_token_fn = target_token_fn
        self.target_labels   = target_labels
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _predict(self, samples):
        enc = self._encode(samples)
        out = self.model(**enc).logits
        probs = torch.softmax(out, dim=-1).cpu().numpy()

        tok_idx = self.target_token_fn(enc)
        tok_probs = probs[:, tok_idx, :]

        if getattr(self, "target_labels", None) is None:
            chosen = tok_probs.max(dim=-1).values[:, None]
        else:
            chosen = tok_probs[:, self.target_labels]
            if chosen.ndim == 1:
                chosen = chosen[:, None]
        if isinstance(chosen, torch.Tensor):
            return chosen.cpu().numpy()
        return chosen

class LimeTextNer(NerAdapter, LimeTextExplainer):
    pass

class LimeLayoutNer(NerAdapter,LimeLayoutExplainer):
    pass

class LimeVisionNer(NerAdapter, LimeVisionExplainer):
    pass