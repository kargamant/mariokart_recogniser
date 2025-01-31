import argparse
import os.path
from time import sleep
import cv2
from environments import CURRENT_DIR, WEIGHTS_DIR
from YOLO import YoloDetector

class YoloStreamer:
    def __init__(self, stream, path_to_model=os.path.join(WEIGHTS_DIR, 'best.pt')):
        self.stream = stream
        self.detector = YoloDetector(path_to_model)

    def process_stream(self, res_dir=''):
        cap = cv2.VideoCapture(self.stream)

        ret, frame = cap.read()
        out_stream = cv2.VideoWriter(res_dir, -1, 20, frame.shape[::-1][1::])
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = self.detector.detect_image(frame)
            if len(res_dir):
                out_stream.write(frame)
            else:
                cv2.imshow('Recognition', res)

            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        out_stream.release()
        cv2.destroyAllWindows()
