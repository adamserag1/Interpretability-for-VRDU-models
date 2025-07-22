"""
Fidelity Metrics for VRDU Models
"""
import torch
from transformers import AutoProcessor, AutoModelForSequenceClassification, AutoModelForTokenClassification
from datasets import load_dataset
from typing import List, Dict, Callable
from tqdm.auto import tqdm
import numpy as np
from vrdu_utils.module_types import DocSample
from vrdu_utils.encoders import make_layoutlmv3_encoder, make_bros_encoder
from Classification_Explain.fidelity import FidelityEvaluator


class ModelWrapper:
    """
    A wrapper for models to provide a consistent prediction interface.
    """
    def __init__(self, model, processor, model_type, device=None):
        self.model = model.eval()
        self.processor = processor
        self.model_type = model_type
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if model_type.lower() == "llmv3":
            self.encode_fn = make_layoutlmv3_encoder(self.processor)
        elif model_type.lower() == "bros":
            self.encode_fn = make_bros_encoder(self.processor)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @torch.no_grad()
    def predict_proba(self, samples):
        """
        Returns prediction probabilities for a list of DocSamples.
        """
        encoded_inputs = self.encode_fn(samples, self.device)
        logits = self.model(**encoded_inputs).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_label(self, sample):
        """
        Returns the predicted label for a single DocSample.
        """
        probs = self.predict_proba([sample])
        return np.argmax(probs, axis=-1)[0]


class EvaluationSuite:
    """
    Main class for running the evaluation suite.
    """
    def __init__(self, model_name: str, model_type: str, task_type: str, device=None):
        self.model_name = model_name
        self.model_type = model_type
        self.task_type = task_type
        self.device = device

        if task_type == "document_classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task_type == "ner":
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model_wrapper = ModelWrapper(self.model, self.processor, self.model_type, self.device)
        self.fidelity_evaluator = FidelityEvaluator(self.model, self.model_wrapper.encode_fn, self.device)

    def load_dataset_for_task(self, dataset_name: str, split: str = "test"):
        """
        Loads and prepares the dataset based on the task type.
        """
        if dataset_name == "funsd" and self.task_type == "ner":
            dataset = load_dataset("nielsr/funsd", split=split)
            # Convert FUNSD dataset to DocSample format for NER
            processed_dataset = []
            for entry in dataset:
                words = [word_info["text"] for word_info in entry["words"]]
                bboxes = [word_info["box"] for word_info in entry["words"]]
                ner_tags = entry["ner_tags"]
                image = entry["image"]
                processed_dataset.append(DocSample(image=image, words=words, bboxes=bboxes, ner_tags=ner_tags, label=None)) # Label is not directly used for NER fidelity
            return processed_dataset
        elif dataset_name == "rvl_cdip" and self.task_type == "document_classification":
            # Assuming RVL-CDIP is already processed into DocSample format or can be easily converted
            # For simplicity, let's assume a dummy load for now. User needs to provide actual RVL-CDIP DocSamples.
            print("Please provide your RVL-CDIP subset as a list of DocSample objects.")

            return [] # Placeholder
        else:
            raise ValueError(f"Unsupported dataset {dataset_name} for task {self.task_type}")

    def evaluate_fidelity(self, samples: List[DocSample], explanations: List[Dict[str, float]], top_k_fraction: float = 0.2) -> List[Dict[str, float]]:
        """
        Evaluates fidelity metrics for a list of samples and their explanations.
        """
        return self.fidelity_evaluator.evaluate_batch(samples, explanations, top_k_fraction)


# Example Usage (to be added in documentation phase)
# if __name__ == "__main__":
#     # Example for LayoutLMv3 on FUNSD NER
#     eval_suite_llmv3_ner = EvaluationSuite(
#         model_name="microsoft/layoutlmv3-base-finetuned-funsd",
#         model_type="layoutlmv3",
#         task_type="ner"
#     )
#     funsd_samples = eval_suite_llmv3_ner.load_dataset_for_task("funsd")

#     # Dummy explanations for demonstration (replace with actual LIME explanations)
#     dummy_explanations = []
#     for sample in funsd_samples:
#         explanation = {word: np.random.rand() for word in sample.words}
#         dummy_explanations.append(explanation)

#     fidelity_results_llmv3_ner = eval_suite_llmv3_ner.evaluate_fidelity(funsd_samples, dummy_explanations)
#     print("LayoutLMv3 FUNSD NER Fidelity Results:", fidelity_results_llmv3_ner[:5])

#     # Example for BROS on RVL-CDIP (conceptual, requires actual RVL-CDIP DocSamples)
#     # eval_suite_bros_doc_cls = EvaluationSuite(
#     #     model_name="path/to/your/bros-finetuned-rvl-cdip", # Replace with your BROS model path
#     #     model_type="bros",
#     #     task_type="document_classification"
#     # )
#     # rvl_cdip_samples = eval_suite_bros_doc_cls.load_dataset_for_task("rvl_cdip") # This will return empty for now

#     # if rvl_cdip_samples:
#     #     dummy_explanations_bros = []
#     #     for sample in rvl_cdip_samples:
#     #         explanation = {word: np.random.rand() for word in sample.words}
#     #         dummy_explanations_bros.append(explanation)
#     #     fidelity_results_bros_doc_cls = eval_suite_bros_doc_cls.evaluate_fidelity(rvl_cdip_samples, dummy_explanations_bros)
#     #     print("BROS RVL-CDIP Document Classification Fidelity Results:", fidelity_results_bros_doc_cls[:5])