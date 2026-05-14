"""
    CLIP text embedding extractor capsule
"""

import os
import cv2
import sys
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.Clip.src.utils.response import build_response_string
from capsules.Clip.src.models.PackageModel import PackageModel
from sdks.novavision.src.base.application import Application


class ClipString(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.inputData = self.request.get_param("inputData")   # string veya string listesi
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

        # String veya liste olabilir, normalize et
        if isinstance(self.inputData, str):
            texts = [self.inputData]
        elif isinstance(self.inputData, list):
            texts = self.inputData

        batch_size = int(self.batchSize) if self.batchSize else 32

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embs = self.clip_model.get_text_features(**inputs)

            all_embeddings.extend(embs.cpu().numpy().tolist())

        # Tek string geldiyse tek embedding döndür
        # Liste geldiyse liste döndür
        if isinstance(self.inputData, str):
            self.outputData = {
                "text":      self.inputData,
                "embedding": all_embeddings[0]
            }
        else:
            self.outputData = [
                {
                    "text":      texts[i],
                    "embedding": all_embeddings[i]
                }
                for i in range(len(texts))
            ]

        self.data = self.outputData
        return build_response_string(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()