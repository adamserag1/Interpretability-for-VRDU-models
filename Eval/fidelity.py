
"""
Fidelity Metrics for VRDU Models
"""

import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Callable

from vrdu_utils.module_types import DocSample


def calculate_comprehensiveness(predict_fn, sample, explanation, mask_token, top_k_fraction= 0.2):
    """
    Calculates the comprehensiveness score for a given explanation.

    Args:
        predict_fn: A function that takes a DocSample and returns the model's prediction probability for the original class.
        sample: The original DocSample.
        explanation: A dictionary where keys are features (e.g., words) and values are their importance scores.
        top_k_fraction: The fraction of top features to remove.

    Returns:
        The comprehensiveness score.
    """
    original_prob = predict_fn(sample)

    # Get top-k features to remove
    sorted_features = sorted(explanation.items(), key=lambda item: item[1], reverse=True)
    top_k = int(len(sorted_features) * top_k_fraction)
    features_to_remove = {item[0] for item in sorted_features[:top_k]}

    # Create perturbed sample by removing top features
    perturbed_words = [word if word not in features_to_remove else mask_token for word in sample.words]
    perturbed_sample = DocSample(image=sample.image, words=perturbed_words, bboxes=sample.bboxes, ner_tags=sample.ner_tags, label=sample.label)

    perturbed_prob = predict_fn(perturbed_sample)

    return original_prob - perturbed_prob


def calculate_sufficiency(predict_fn, sample, explanation, mask_token, top_k_fraction= 0.2):
    """
    Calculates the sufficiency score for a given explanation.

    Args:
        predict_fn: A function that takes a DocSample and returns the model's prediction probability for the original class.
        sample: The original DocSample.
        explanation: A dictionary where keys are features (e.g., words) and values are their importance scores.
        top_k_fraction: The fraction of top features to keep.

    Returns:
        The sufficiency score.
    """
    original_prob = predict_fn(sample)
    print(explanation.items())
    # Get top-k features to keep
    sorted_features = sorted(explanation.items(), key=lambda item: item[1], reverse=True)
    print(sorted_features)
    top_k = int(len(sorted_features) * top_k_fraction)

    features_to_keep = {item[0] for item in sorted_features[:top_k]}
    print(features_to_keep)

    # Create perturbed sample by keeping only top features
    perturbed_words = [word if word in features_to_keep else mask_token for word in sample.words]
    # ALIGN BBOXES
    print(perturbed_words)
    perturbed_sample = DocSample(image=sample.image, words=perturbed_words, bboxes=sample.bboxes, ner_tags=sample.ner_tags, label=sample.label)

    perturbed_prob = predict_fn(perturbed_sample)
    print(f'original probability: {original_prob}, pertrubed_probability: {perturbed_prob}')
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