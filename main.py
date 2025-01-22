import cv2
from Haar import DataPreparator, Trainer, Detector, Visualizer
import os
from environments import *


if __name__ == '__main__':
    # todo: choice of Haar vs YOLO with arguments from cmd
    # todo: also make an option to choose between boo and coins recognition

    # full pipeline
    dp = DataPreparator(os.path.join(CURRENT_DIR, BOO_DIR), os.path.join(CURRENT_DIR, GP_DIR))
    dp.prepare_labels()

    Trainer.create_vec_file(GOOD_FILE_DIR, VEC_FILE_DIR, 30, 30)
    Trainer.train_cascade(MODEL_DIR, VEC_FILE_DIR, BAD_FILE_DIR, 10, 10, 10, 30, 30)

    detector = Detector(os.path.join(CURRENT_DIR, MODEL_DIR, 'cascade.xml'))

    test_results = []
    for img in os.listdir(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images')):
        test_results.append(detector.detect(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images', str(img))))

    visualizer = Visualizer(test_results)
    visualizer.viz()
