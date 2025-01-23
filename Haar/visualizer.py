import cv2
import os
from time import sleep

class Visualizer:

    @classmethod
    def viz(cls, img_list, write=False, save_dir=''):
        i = 0
        for img in img_list:
            if write:
                cv2.imwrite(os.path.join(save_dir, f'{i}.jpg'), img)
                i += 1
            else:
                sleep(1)
                cv2.imshow('image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
