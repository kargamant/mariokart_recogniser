import torch
import numpy as np


class YoloDetector:
    def __init__(self, weights_path: str):
        # but mb just torch.load should work too
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

    def detect(self, path_to_image: str):
        res = self.model(path_to_image)
        res = np.squeeze(res.render())
        return res

    def detect_image(self, image):
        res = self.model(image)
        res = np.squeeze(res.render())
        return res