import argparse
from VideoStream import YoloStreamer, HaarStreamer
from environments import CURRENT_DIR
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-yolo', '--yolo', action='store_true', help='uses yolo for video stream recognition.')
    parser.add_argument('-haar', '--haar', action='store_true', help='uses haar for video stream recognition.')
    parser.add_argument('-st', '--stream', type=str, default=os.path.join(CURRENT_DIR, 'mk_boo.mp4'), help='stream to get frames from')
    parser.add_argument('-res', '--results_dir', type=str, default='', help='where to save video')

    args = parser.parse_args()
    if args.yolo:
        try:
            args.stream = int(args.stream)
        except ValueError:
            pass
        yolo = YoloStreamer(stream=args.stream)
        yolo.process_stream(args.results_dir)
    elif args.haar:
        try:
            args.stream = int(args.stream)
        except ValueError:
            pass
        haar = HaarStreamer(stream=args.stream)
        haar.process_stream(args.results_dir)

    print('Bye!')

