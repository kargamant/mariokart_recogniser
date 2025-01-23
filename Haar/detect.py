import cv2


class Detector:
    def __init__(self, path_to_model: str):
        self.model = cv2.CascadeClassifier(path_to_model)

    def detect(self, path_to_image: str, minNeighbors = 20, minSize = (70, 70)):
        image = cv2.imread(path_to_image)
        obj = self.model.detectMultiScale(image, minNeighbors=minNeighbors, minSize=minSize)
        for (x, y, w, h) in obj:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 5)
        return image

    def detect_image(self, image, minNeighbors = 20, minSize = (70, 70)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        obj = self.model.detectMultiScale(gray, minNeighbors=minNeighbors, minSize=minSize)
        for (x, y, w, h) in obj:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
        return image
