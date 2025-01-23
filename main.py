import cv2
from Haar import DataPreparator, Trainer, Detector, Visualizer
import os
from environments import *
import sys


if __name__ == '__main__':
    # todo: choice of Haar vs YOLO with arguments from cmd
    # todo: also make an option to choose between boo and coins recognition

    options = []
    for i in range(1, len(sys.argv)):
        options.append(sys.argv[i])

    # full pipeline
    if '-prep' in options or 'all' in options:
        dp = DataPreparator(os.path.join(CURRENT_DIR, BOO_DIR), os.path.join(CURRENT_DIR, GP_DIR))
        dp.prepare_labels()

    if '-train' in options or 'all' in options:
        Trainer.create_vec_file(GOOD_FILE_DIR, VEC_FILE_DIR, 70, 70)
        Trainer.train_cascade(MODEL_DIR, VEC_FILE_DIR, BAD_FILE_DIR, 70, 50, 5, 70, 70)

    detector = Detector(os.path.join(CURRENT_DIR, MODEL_DIR, 'cascade.xml'))

    test_results = []
    for img in os.listdir(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images'))[:20:]:
          test_results.append(detector.detect(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images', str(img)), 20, (70, 70)))
          print(img)
    Visualizer.viz(test_results, write=0, save_dir='results')
