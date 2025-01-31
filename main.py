import cv2
from Haar import DataPreparator, Trainer, Detector, Visualizer
from YOLO import YoloDetector
import os
from environments import *
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # todo: choice of Haar vs YOLO with arguments from cmd
    # todo: also make an option to choose between boo and coins recognition

    parser = argparse.ArgumentParser()
    parser.add_argument('-yolo', '--yolo', action='store_true', help='chooses yolo over haar cascades to perform command') # these are for testing mode
    parser.add_argument('-haar', '--haar', action='store_true', help='chooses haar over yolo to perform command')
    parser.add_argument('-prep', '--preparator', action='store_true', help='turns on data preparation from original dataset')
    parser.add_argument('-train', '--trainer', action='store_true', help='trains haar cascade. yolo training should be done separately')
    parser.add_argument('-all', '--all', action='store_true', help='prep+train+test for haar cascade')
    parser.add_argument('-one_file', '--one_file', help='feeding one file to get output with bounding box')
    parser.add_argument('-w', '--width', type=int, default=70, help='minimum width')
    parser.add_argument('-he', '--height', type=int, default=70, help='minimum height')
    parser.add_argument('-pos', '--positives', type=int, default=70, help='number of positives to use during training haar')
    parser.add_argument('-neg', '--negatives', type=int, default=50, help='number of negatives to use during training haar')
    parser.add_argument('-st', '--stages', type=int, default=5, help='number of stages to train haar')
    parser.add_argument('-mnb', '--minNeighbours', type=int, default=20, help='min neighbours to recognise with haar')
    parser.add_argument('-test', '--test', action='store_true', help='test mode')
    parser.add_argument('-nimg', '--nimages', type=int, default=30, help='how many images to test from test set')
    parser.add_argument('-res', '--results_dir', type=str, default='', help='test results directory. If not specified then results will be shown in window.')



    args = parser.parse_args()

    # full pipeline
    if args.preparator or args.all:
        dp = DataPreparator(os.path.join(CURRENT_DIR, BOO_DIR), os.path.join(CURRENT_DIR, GP_DIR))
        dp.prepare_labels()

    if args.trainer or args.all:
        # todo automatically clean data dir
        Trainer.create_vec_file(GOOD_FILE_DIR, VEC_FILE_DIR, args.width, args.height)
        Trainer.train_cascade(MODEL_DIR, VEC_FILE_DIR, BAD_FILE_DIR, args.positives, args.negatives, args.stages, args.width, args.height)

    detector = Detector(os.path.join(CURRENT_DIR, MODEL_DIR, 'cascade.xml'))

    if args.one_file and args.haar:
        result = detector.detect(args.one_file, args.minNeighbours, (args.width, args.height))
        Visualizer.viz([result])
    elif args.one_file and args.yolo:
        detector_yolo = YoloDetector(os.path.join(WEIGHTS_DIR, 'best.pt'))
        res = detector_yolo.detect(args.one_file)
        Visualizer.viz([res])
    elif args.test and args.haar:
        # testing
        print('If you want to quit the view mode and stop just hit q button')
        for img in os.listdir(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images'))[:args.nimages:]:
              res = detector.detect(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images', str(img)), args.minNeighbours, (args.width, args.height))
              print(img)
              if len(args.results_dir):
                  plt.imsave(os.path.join(args.results_dir, f'{img.replace(".jpg", "")}_test.jpg'), res)
              else:
                key_code = Visualizer.viz([res])
                if key_code == ord('q'):
                    break
    elif args.test and args.yolo:
        detector_yolo = YoloDetector(os.path.join(WEIGHTS_DIR, 'best.pt'))
        print('If you want to quit the view mode and stop just hit q button')
        for img in os.listdir(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images')):
            res = detector_yolo.detect(os.path.join(CURRENT_DIR, BOO_DIR, 'test', 'images', str(img)))
            print(img)
            if len(args.results_dir):
                plt.imsave(os.path.join(args.results_dir, f'{img.replace(".jpg", "")}_test.jpg'), res)
            else:
                key_code = Visualizer.viz([res])
                if key_code == ord('q'):
                    break

    print('Bye!\n')


