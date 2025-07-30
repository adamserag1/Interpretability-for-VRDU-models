import torch
from PIL import Image
from pandas.io.formats.format import return_docstring
from vrdu_utils.utils import normalize_bbox

def _stack_on_decive(enc, device):
    return {k: v.to(device) for k, v in enc.items()}

def make_layoutlmv3_encoder(processor, ner = False, max_length: int = 512):
    """"
    LayoutLMv3 encoder for document classification.
    """
    def encode(samples, device):
        images = [s.image.convert("RGB") for s in samples]
        words = [s.words for s in samples]
        ner_tags = [s.ner_tags for s in samples]

        boxes = []

        if not ner:
            enc = processor(
                images,
                words,
                boxes=boxes,
                truncation=True,
                padding = "max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            for s in samples:
                w, h = s.image.size
                boxes.append([normalize_bbox(b, w, h) for b in s.bboxes])
            enc = processor(
                images,
                words,
                boxes=boxes,
                word_labels=ner_tags,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        if "bbox" in enc:
            enc["bbox"] = enc["bbox"].clamp_(0, 1000)

        return _stack_on_decive(enc, device)

    return encode


def make_bros_encoder(tokenizer, ner = False, max_length = 512):
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
        if ner:
            batch_normalized_bboxes, encoded_labels = [], []
            # print(samples.bboxes, samples.image_path, samples.ner_tags) ## DEBUGGING

            for idx, s in enumerate(samples):
                width, height = s.image.size
                #normalized_bboxes = [normalize_bbox(bbox, width, height) for bbox in s.bboxes]

                # Align boxes to sub words
                aligned_boxes, aligned_labels = [], []
                for word_id in enc.word_ids(batch_index=idx):
                    if word_id is None:
                        aligned_boxes.append([0, 0, 0, 0])
                        aligned_labels.append(-100)
                    else:
                        aligned_boxes.append(s.bboxes[word_id])
                        aligned_labels.append(s.ner_tags[word_id])

                batch_normalized_bboxes.append(aligned_boxes)
                encoded_labels.append(aligned_labels)

            enc['bbox'] = torch.tensor(batch_normalized_bboxes, dtype=torch.float)
            enc['labels'] =torch.tensor(encoded_labels, dtype=torch.long)
        else:
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
