import torch
from transformers import CLIPModel, CLIPProcessor
from sdks.novavision.src.base.application import Application


class ModelLoader:
    def __init__(self, config: dict):
        self.config      = config
        self.application = Application()

    def load_model(self) -> dict:
        model_name = self.application.get_param(config=self.config, name="modelName")
        device_cfg  = self.application.get_param(config=self.config, name="device") or "cpu"

        device = "cuda:0" if device_cfg == "GPU" and torch.cuda.is_available() else "cpu"

        clip_model = CLIPModel.from_pretrained(model_name)
        processor  = CLIPProcessor.from_pretrained(model_name)
        clip_model = clip_model.to(device).eval()

        print(f"[CLIP] Model yüklendi: {model_name} | device: {device}")

        return {
            "clip_model": clip_model,
            "processor":  processor,
            "device":     device,
        }