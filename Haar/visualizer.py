import cv2

class Visualizer:
    def __init__(self, img_list):
        self.img_list = img_list

    def viz(self):
        for img in self.img_list:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
