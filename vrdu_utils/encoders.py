import torch
from PIL import Image
from pandas.io.formats.format import return_docstring
from vrdu_utils.utils import normalize_bbox

def _stack_on_decive(enc, device):
    return {k: v.to(device) for k, v in enc.items()}

def make_layoutlmv3_encoder_cls(processor, max_length: int = 128):
    """"
    LayoutLMv3 encoder for document classification.
    """
    def encode(samples, device):
        images = [s.image.convert("RGB") for s in samples]
        words = [s.words for s in samples]

        boxes = []
        for s in samples:
            w, h = s.image.size
            boxes.append([normalize_bbox(b, w, h) for b in s.bboxes])

        enc = processor(
            images,
            words,
            boxes=boxes,
            truncation=True,
            padding = "max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return _stack_on_decive(enc, device)

    return encode


def make_bros_encoder_cls(tokenizer, max_length = 512):
    """
    BROS encoder for document classification.
    """
    def encode(samples, device):
        words = [s.words for s in samples]

        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding = "max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        aligned_boxes = []
        for idx, s in enumerate(samples):
            w, h = s.image.size
            norm = [normalize_bbox(b, w, h) for b in s.bboxes]
            word_ids = enc.word_ids(batch_index=idx)
            aligned_boxes.append(
                [norm[wid] if wid is not None else [0,0,0,0] for wid in word_ids]
            )
        enc["bbox"] = torch.tensor(aligned_boxes, dtype=torch.long)
        return _stack_on_decive(enc, device)
    return encode

# TODO: Add token classification encoders
