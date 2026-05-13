"""
    CLIP embedding extractor capsule
"""

import os
import cv2
import sys
import numpy as np
from PIL import Image as PILImage
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.Clip.src.utils.response import build_response
from capsules.Clip.src.models.PackageModel import PackageModel
from sdks.novavision.src.base.application import Application


class Clip(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.images    = self.request.get_param("inputImage")   # liste geliyor
        self.batchSize = self.request.get_param("batchSize")

        self.clip_model = bootstrap["clip_model"]
        self.processor  = bootstrap["processor"]
        self.device     = bootstrap["device"]

        self.outputData = []
        print("init")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        application = Application()

        model_name = application.get_param(config=config, name="modelName")
        device = application.get_param(config=config, name="device")

        clip_model = CLIPModel.from_pretrained(model_name)
        processor  = CLIPProcessor.from_pretrained(model_name)
        clip_model = clip_model.to(device).eval()

        print("bootstrap")
        return {
            "clip_model": clip_model,
            "processor":  processor,
            "device":     device,
        }

    def run(self):
        print("run")

        if not self.images:
            self.data = []
            return build_response(context=self)

        batch_size = int(self.batchSize) if self.batchSize else 32

        # ── Tüm görüntüleri oku, uID'leri sakla ──
        pil_images = []
        uids       = []

        for img in self.images:
            uid   = img.get("uID", "")
            frame = Image.get_frame(img=img, redis_db=self.redis_db)

            if frame is None:
                pil_images.append(PILImage.new("RGB", (224, 224)))
            else:
                img_np = np.asarray(frame.value).astype(np.uint8)
                pil_images.append(PILImage.fromarray(img_np[..., ::-1]))

            uids.append(uid)

        # ── Batch'lere böl, embedding hesapla ──
        all_embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch  = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
                embs    = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output

            all_embeddings.extend(embs.cpu().numpy().tolist())

        # ── uID + embedding eşleştir ──
        self.outputData = [
            {
                "uID":       uids[i],
                "embedding": all_embeddings[i]
            }
            for i in range(len(uids))
        ]

        self.data = self.outputData
        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()