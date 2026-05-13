"""
    ...
"""

import os
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.Clip.src.utils.response import build_response
from capsules.Clip.src.models.PackageModel import PackageModel


class Clip(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.batchsize = self.request.get_param("batchSize")
        print("init")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        print("bootstrap")
        return {}

    def run(self):
        print("run")

        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()