"""
Applying Lime w.r.t VRDU modalities 
"""

class BaseLimeExplainer:
    def __init__(self, model, processor, device: torch.devce):
        self.model = model.eval
        self.processor = processor
        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.class_names = [v for k, v in sorted(model.config.id2label.items())]
    
    def _encode(self, samples: Sequence[DocSample]) -> Dict[str, torch.Tensor]:
        images = [s.image for s in samples]
        texts = [s.words for s in samples]
        boxes = [s.boxes for s in samples]
        try:
            enc = self.processor(images = images,
                                texts = texts,
                                boxes = boxes,
                                return_tensors="pt",
                                padding = "max_length",
                                truncation = True,
                                max_length=512)
        except:
            enc = self.processor(texts = texts,
                                boxes = boxes,
                                return_tensors = "pt",
                                padding = "max_length",
                                truncation = True,
                                max_length = 512)
        

    