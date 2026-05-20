import os
import torch
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from sdks.novavision.src.base.application import Application
from sdks.novavision.src.base.logger import LoggerManager

logger = LoggerManager()

STORAGE_DIR = "/storage/clip"


class ModelLoader:
    def __init__(self, config: dict):
        self.config      = config
        self.application = Application()

    def load_model(self) -> dict:
        model_name = self.application.get_param(config=self.config, name="modelName")
        device_cfg  = self.application.get_param(config=self.config, name="device") or "cpu"
        device = "cuda:0" if device_cfg == "GPU" and torch.cuda.is_available() else "cpu"

        model_path = download_clip(model_name)

        clip_model = CLIPModel.from_pretrained(model_path)
        processor  = CLIPProcessor.from_pretrained(model_path)
        clip_model = clip_model.to(device).eval()

        print(f"[CLIP] Model yüklendi: {model_name} | device: {device}")

        return {
            "clip_model": clip_model,
            "processor":  processor,
            "device":     device,
        }


def download_clip(model_name: str, storage_dir: str = STORAGE_DIR) -> str:
    """
    Model daha önce /storage/clip altına indirildiyse oradan yükler.
    İndirilmediyse Hugging Face'den indirip /storage/clip altına kaydeder.
    """
    try:
        # model adındaki / karakterini _ ile değiştir (klasör adı için)
        safe_name  = model_name.replace("/", "_")
        model_path = Path(storage_dir) / safe_name

        if model_path.exists():
            print(f"[CLIP] Model cache'den yükleniyor: {model_path}")
            return str(model_path)

        print(f"[CLIP] Model indiriliyor: {model_name}")
        model_path.mkdir(parents=True, exist_ok=True)

        clip_model = CLIPModel.from_pretrained(model_name)
        processor  = CLIPProcessor.from_pretrained(model_name)

        clip_model.save_pretrained(str(model_path))
        processor.save_pretrained(str(model_path))

        print(f"[CLIP] Model kaydedildi: {model_path}")
        return str(model_path)

    except Exception as e:
        print(e)
        logger.error(f"CLIP - Model indirilemedi: {model_name}")
        raise