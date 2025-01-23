import os
from YOLO import YoloDetector
from environments import WEIGHTS_DIR, CURRENT_DIR, BOO_DIR
import matplotlib.pyplot as plt
from time import sleep


if __name__ == '__main__':
    dt = YoloDetector(os.path.join(WEIGHTS_DIR, 'exp5', 'weights', 'best.pt'))
    for img in os.listdir(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images')):
        res = dt.detect(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images', str(img)))
        plt.imsave('yolo_boo.jpg', res)
        break
