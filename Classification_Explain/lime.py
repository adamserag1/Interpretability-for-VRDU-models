"""
Applying Lime w.r.t VRDU modalities 
"""

import numpy as np
import torch
from lime.lime_tabular import LimeTabularExplainer

from vrdu_utils.module_types import DocSample


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
    def __init__(self, model, encode_fn, mask_token = "[UNK]", device=None, kernel_width_factor=1.0):
        super().__init__(model, encode_fn, device)
        self.mask_token = mask_token
        self.kernel_width_factor = kernel_width_factor

    def _make_predict_fn(self, sample: DocSample):
        def fn(z_bin_list):
            perturbed = []
            for z in z_bin_list:
                words = [w if z_i else self.mask_token for w, z_i in zip(sample["words"], z)]
               # boxes = [b if z_i else [0,0,0,0] for b, z_i in zip(sample.bboxes, z)] # change to height width of image
                boxes = sample["bboxes"]
                perturbed.append(DocSample(sample["image"], words, boxes))
            return self._predict(perturbed)
        print("MADE PREDICT")
        return fn

    def explain(self, sample: DocSample, num_samples = 4000, num_features=30):
        n_tokens = len(sample["words"])
        explainer = LimeTabularExplainer(
            training_data = np.vstack([np.ones(n_tokens), np.zeros(n_tokens)]),
            feature_names = sample["words"],
            class_names = self.class_names,
            discretize_continuous = False,
            categorical_features = list(range(n_tokens)),
            mode = 'classification',
            kernel_width = np.sqrt(n_tokens) * self.kernel_width_factor
        )
        print("Begging EXPLAIN_INSTANCE")
        return explainer.explain_instance(
            data_row = np.ones(n_tokens),
            predict_fn=self._make_predict_fn(sample),
            num_samples = num_samples,
            num_features = num_features
        )
