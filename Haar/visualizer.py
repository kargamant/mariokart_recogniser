import cv2
import os
from time import sleep

class Visualizer:

    @classmethod
    def viz(cls, img_list):
        key_code = 0
        for img in img_list:
            sleep(1)
            cv2.imshow('image', img)
            key_code = cv2.waitKey(0)
            cv2.destroyAllWindows()
        return key_code
