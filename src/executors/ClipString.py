"""
    CLIP text embedding extractor capsule
"""

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.Clip.src.utils.response import build_response_string
from capsules.Clip.src.utils.utils import ModelLoader
from capsules.Clip.src.models.PackageModel import PackageModel


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
        return ModelLoader(config=config).load_model()

    def run(self):
        print("run string")

        if isinstance(self.inputData, str):
            texts = [self.inputData]
        elif isinstance(self.inputData, list):
            texts = self.inputData
        else:
            self.data = []
            return build_response_string(context=self)

        batch_size = int(self.batchSize) if self.batchSize else 32

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model.get_text_features(**inputs)
                embs    = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output

            all_embeddings.extend(embs.detach().clone().cpu().numpy().tolist())

        if isinstance(self.inputData, str):
            self.outputData = {
                "text":      self.inputData,
                "embedding": all_embeddings[0]
            }
        else:
            self.outputData = [
                {"text": texts[i], "embedding": all_embeddings[i]}
                for i in range(len(texts))
            ]

        self.data = self.outputData
        return build_response_string(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()