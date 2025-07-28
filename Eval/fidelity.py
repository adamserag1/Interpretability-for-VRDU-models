
"""
Fidelity Metrics for VRDU Models
"""

import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Callable

from vrdu_utils.module_types import DocSample

def _top_k_indices(sorted_features, top_k):
    """Return the *indices* of the top-k layout boxes."""
    return {i for i, (_box, _score) in enumerate(sorted_features[:top_k])}

def calculate_comprehensiveness(predict_fn, sample, explanation, mask_token, top_k=5, modality='text'):
    """
    Calculates the comprehensiveness score for a given explanation.

    Args:
        predict_fn: A function that takes a DocSample and returns the model's prediction probability for the original class.
        sample: The original DocSample.
        explanation: A dictionary where keys are features (e.g., words) and values are their importance scores.
        top_k: The number of top features to remove.

    Returns:
        The comprehensiveness score.
    """
    original_prob = predict_fn(sample)

    # Get top-k features to remove
    sorted_features = sorted(explanation.items(), key=lambda item: item[1], reverse=True)
    top_k = min(top_k, len(sorted_features))
    w, h = sample.image.size
    if modality == 'text':
        remove_set = {kv[0] for kv in sorted_features[:top_k]}
        words = [w if w not in remove_set else mask_token
                 for w in sample.words]
        bboxes = sample.bboxes

    if modality == 'layout':
        remove_idx = _top_k_indices(sorted_features, top_k)

        bboxes = [
            bbox if i not in remove_idx else [0, 0, *sample.image.size]
            for i, bbox in enumerate(sample.bboxes)
        ]
        words = sample.words
    if modality == 'vision':
            print('VISION NOT IMPLEMENTED')

    print(f'Removed top {top_k} words')
    perturbed_sample = DocSample(sample.image, words, bboxes,
                          ner_tags=sample.ner_tags, label=sample.label)
    perturbed_prob = predict_fn(perturbed_sample)
    print(f'[COMP] original probability: {original_prob}, perturbed_probability: {perturbed_prob}')
    return original_prob - perturbed_prob


def calculate_sufficiency(predict_fn, sample, explanation, mask_token, top_k=5, modality='text'):
    """
    Calculates the sufficiency score for a given explanation.

    Args:
        predict_fn: A function that takes a DocSample and returns the model's prediction probability for the original class.
        sample: The original DocSample.
        explanation: A dictionary where keys are features (e.g., words) and values are their importance scores.
        top_k: The number of top features to keep.

    Returns:
        The sufficiency score.
    """
    original_prob = predict_fn(sample)
    # Get top-k features to keep
    sorted_features = sorted(explanation.items(), key=lambda item: item[1], reverse=True)
    top_k = min(top_k, len(sorted_features))

    # Create perturbed sample by keeping only top features
    w, h = sample.image.size
    if modality == 'text':
        keep_set = {kv[0] for kv in sorted_features[:top_k]}
        words = [w if w in keep_set else mask_token
                 for w in sample.words]
        bboxes = sample.bboxes
    if modality == 'layout':
        keep_idx = _top_k_indices(sorted_features, top_k)

        bboxes = [
            bbox if i in keep_idx else [0, 0, *sample.image.size]
            for i, bbox in enumerate(sample.bboxes)
        ]
        words = sample.words
    if modality == 'vision':
        print('VISION NOT IMPLEMENTED')
    perturbed_sample = DocSample(image=sample.image, words=words, bboxes=bboxes, ner_tags=sample.ner_tags, label=sample.label)

    perturbed_prob = predict_fn(perturbed_sample)
    print(f'Kept top {top_k} {modality} tokens')
    print(f'[SUF] original probability: {original_prob}, perturbed_probability: {perturbed_prob}')
    return original_prob - perturbed_prob


class FidelityEvaluator:
    """
    A class to evaluate the fidelity of explanations for a given model.
    """
    def __init__(self, model, encode_fn, mask_token = "[UNK]", device=None):
        self.model = model.eval()
        self.encode_fn = encode_fn
        self.mask_token = mask_token
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_prediction_function(self, original_label_index):
        """
        Returns a function that predicts the probability of the original label for a given sample.
        """
        @torch.no_grad()
        def predict_fn(sample: DocSample):
            encoded = self.encode_fn([sample], self.device)
            try:
                logits = self.model(**encoded).logits
            except:
                logits = self.model(**encoded)["logits"]
            probs = torch.softmax(logits, dim=-1)
            return probs[0, original_label_index].item()
        return predict_fn

    def evaluate(self, sample: DocSample, explanation, top_k_fraction= 0.2):
        """
        Evaluates the fidelity of an explanation for a single sample.

        Args:
            sample: The DocSample to evaluate.
            explanation: The explanation for the sample.
            top_k_fraction: The fraction of features to use for comprehensiveness and sufficiency.

        Returns:
            A dictionary with the comprehensiveness and sufficiency scores.
        """
        original_label_index = sample.label
        predict_fn = self._get_prediction_function(original_label_index)

        comprehensiveness = calculate_comprehensiveness(predict_fn, sample, explanation, self.mask_token, top_k_fraction)
        sufficiency = calculate_sufficiency(predict_fn, sample, explanation, self.mask_token, top_k_fraction)

        return {
            "comprehensiveness": comprehensiveness,
            "sufficiency": sufficiency
        }

    def evaluate_batch(self, samples, explanations, top_k_fraction= 0.2):
        """
        Evaluates the fidelity of explanations for a batch of samples.
        """
        results = []
        for sample, explanation in tqdm(zip(samples, explanations), total=len(samples), desc="Evaluating Fidelity"):
            results.append(self.evaluate(sample, explanation, top_k_fraction))
        return results