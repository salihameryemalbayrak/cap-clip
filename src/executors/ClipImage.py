"""
    CLIP image embedding extractor capsule
"""

import os
import sys
import numpy as np
from PIL import Image as PILImage
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.Clip.src.utils.response import build_response_image
from capsules.Clip.src.utils.utils import ModelLoader
from capsules.Clip.src.models.PackageModel import PackageModel


class ClipImage(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.images    = self.request.get_param("inputImage")
        self.batchSize = self.request.get_param("batchSize")

        self.clip_model = bootstrap["clip_model"]
        self.processor  = bootstrap["processor"]
        self.device     = bootstrap["device"]

        self.outputData = []
        print("init image")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return ModelLoader(config=config).load_model()

    def run(self):
        print("run image")

        if isinstance(self.images, dict):
            self.images = [self.images]

        batch_size = int(self.batchSize) if self.batchSize else 32

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

        all_embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch  = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
                embs    = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output

            all_embeddings.extend(embs.detach().clone().cpu().numpy().tolist())

        self.outputData = [
            {"uID": uids[i], "embedding": all_embeddings[i]}
            for i in range(len(uids))
        ]

        self.data = self.outputData
        return build_response_image(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()