import cv2
from Haar import DataPreparator, Trainer, Detector, Visualizer
import os
from environments import *
import argparse


if __name__ == '__main__':
    # todo: choice of Haar vs YOLO with arguments from cmd
    # todo: also make an option to choose between boo and coins recognition

    parser = argparse.ArgumentParser()
    parser.add_argument('-prep', '--preparator', action='store_true')
    parser.add_argument('-train', '--trainer', action='store_true')
    parser.add_argument('-all', '--all', action='store_true')
    parser.add_argument('-one_file', '--one_file')
    parser.add_argument('-w', '--width', type=int, default=70)
    parser.add_argument('-he', '--height', type=int, default=70)
    parser.add_argument('-pos', '--positives', type=int, default=70)
    parser.add_argument('-neg', '--negatives', type=int, default=50)
    parser.add_argument('-st', '--stages', type=int, default=5)
    parser.add_argument('-mnb', '--minNeighbours', type=int, default=20)
    parser.add_argument('-test', '--test', action='store_true')
    parser.add_argument('-nimg', '--nimages', type=int, default=30)
    parser.add_argument('-write', '--write_res', action='store_true')
    parser.add_argument('-res', '--results_dir', type=str, default='results')


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

    if args.one_file:
        result = detector.detect(args.one_file, args.minNeighbours, (args.width, args.height))
        Visualizer.viz([result])
    elif args.test:
        # testing
        test_results = []
        for img in os.listdir(os.path.join(CURRENT_DIR, BOO_DIR, 'valid', 'images'))[:args.nimages:]:
              test_results.append(detector.detect(os.path.join(CURRENT_DIR, BOO_DIR, 'valid', 'images', str(img)), args.minNeighbours, (args.width, args.height)))
              print(img)
        Visualizer.viz(test_results, write=args.write_res, save_dir=args.results_dir)

    # Just experiment
    # img74 = cv2.imread(os.path.join(CURRENT_DIR, BOO_DIR, 'train', 'images', '267_png.rf.1febe6b0d1325197a730a0d186c38610.jpg'))
    # h, w, c = img74.shape
    # with open(os.path.join(CURRENT_DIR, BOO_DIR, 'train', 'labels', '267_png.rf.1febe6b0d1325197a730a0d186c38610.txt'), 'r') as file:
    #     cl, x, y, w_bb, h_bb = list(map(float, file.readline().split(' ')))
    #
    # x *= w
    # y *= h
    # w_bb *= w
    # h_bb *= h
    #
    # cv2.rectangle(img74, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 5)
    # cv2.imshow('img', img74)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('Bye!\n')


