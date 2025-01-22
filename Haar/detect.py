import cv2


class Detector:
    def __init__(self, path_to_model: str):
        self.model = cv2.CascadeClassifier(path_to_model)

    def detect(self, path_to_image: str):
        image = cv2.imread(path_to_image)
        obj = self.model.detectMultiScale(image)
        for (x, y, w, h) in obj:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 5)
        return image
