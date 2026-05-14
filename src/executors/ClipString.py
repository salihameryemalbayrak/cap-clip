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
        # inputData: string veya list[string] bekliyoruz
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

    def run(self):
        print("run string")

        # ── SADECE STRING VE LIST KONTROLÜ ──
        # dict (sözlük) gelirse işlem yapma
        if isinstance(self.inputData, dict):
            print("Dict input ignored as requested.")
            return build_response_string(context=self)

        if isinstance(self.inputData, list):
            texts_to_process = [str(t) for t in self.inputData]
            is_list = True
        elif isinstance(self.inputData, str):
            texts_to_process = [self.inputData]
            is_list = False
        else:
            # Diğer tipleri (int, None vb.) ignore et
            return build_response_string(context=self)

        batch_size = int(self.batchSize) if self.batchSize else 32
        all_embeddings = []

        for i in range(0, len(texts_to_process), batch_size):
            batch = texts_to_process[i:i + batch_size]

            # Modelin çökmemesi için boş metinleri boşlukla değiştir
            batch = [t if (t and t.strip()) else " " for t in batch]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)

                # Tensor tipini doğrula (BaseModelOutput hatasını önler)
                if isinstance(outputs, torch.Tensor):
                    embs = outputs
                else:
                    embs = getattr(outputs, "text_embeds", outputs[0])

            # Ham embeddingleri listeye ekle (Normalizasyon yok)
            all_embeddings.extend(embs.cpu().numpy().tolist())

        # ── ÇIKTI FORMATLAMA ──
        formatted_results = [
            {
                "value":     texts_to_process[j],
                "embedding": all_embeddings[j]
            }
            for j in range(len(texts_to_process))
        ]

        # Liste geldiyse liste, string geldiyse tek obje dön
        if is_list:
            self.outputData = formatted_results
        else:
            self.outputData = formatted_results[0]

        return build_response_string(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()