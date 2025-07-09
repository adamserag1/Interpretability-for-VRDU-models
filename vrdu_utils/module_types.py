from dataclasses import dataclass
from PIL import Image

@dataclass
class DocSample:
    """
    Container for document page
    """
    image: Image.Image
    words: list
    bboxes: list
    label: int | None = None
    ner_tags: list | None = None


