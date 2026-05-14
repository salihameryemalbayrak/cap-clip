"""
    CLIP text embedding extractor capsule
"""

import os
import sys
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.Clip.src.utils.response import build_response_string
from capsules.Clip.src.models.PackageModel import PackageModel
from sdks.novavision.src.base.application import Application


class ClipString(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.inputData = self.request.get_param("inputData")
        self.batchSize = self.request.get_param("batchSize")

        self.clip_model = bootstrap["clip_model"]
        self.processor  = bootstrap["processor"]
        self.device     = bootstrap["device"]

        self.outputData = []
        print("init string")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        application = Application()
        model_name = application.get_param(config=config, name="modelName")
        device     = application.get_param(config=config, name="device")

        clip_model = CLIPModel.from_pretrained(model_name)
        processor  = CLIPProcessor.from_pretrained(model_name)
        clip_model = clip_model.to(device).eval()

        print("bootstrap string")
        return {
            "clip_model": clip_model,
            "processor":  processor,
            "device":     device,
        }

    def _extract_text(self, data):
        """Farklı JSON formatlarından ham metni ayıklar."""
        if isinstance(data, dict):
            val = data.get("value", "")
            if isinstance(val, dict):
                return str(next(iter(val.values())))
            return val
        return data

    def run(self):
        print("run string")

        # Girdi verisini temizle
        raw_input = self._extract_text(self.inputData)

        # İşleme listesi oluştur
        if isinstance(raw_input, list):
            texts_to_process = [str(t) for t in raw_input]
        else:
            texts_to_process = [str(raw_input)]

        batch_size = int(self.batchSize) if self.batchSize else 32
        all_embeddings = []

        for i in range(0, len(texts_to_process), batch_size):
            batch = texts_to_process[i:i + batch_size]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)

                # 'BaseModelOutputWithPooling' hatasını önlemek için kontrol
                if isinstance(outputs, torch.Tensor):
                    embs = outputs
                else:
                    # pooling_output veya ilk indeksi al
                    embs = getattr(outputs, "text_embeds", outputs[0])

            # İstediğin üzerine normalizasyon (embs / embs.norm) kaldırıldı
            all_embeddings.extend(embs.cpu().numpy().tolist())

        # ClipImage yapısında olduğu gibi listeleme yapıyoruz
        self.outputData = [
            {
                "value":     texts_to_process[j],
                "embedding": all_embeddings[j]
            }
            for j in range(len(texts_to_process))
        ]

        # Eğer tek bir string geldiyse liste yerine direkt objeyi döndürmek istersen:
        if not isinstance(raw_input, list):
            self.data = self.outputData[0]
        else:
            self.data = self.outputData

        return build_response_string(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()