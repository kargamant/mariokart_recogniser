import os.path
from time import sleep
import cv2
from environments import MK_GAMEPLAY, SCREEN_TIME, GP_DIR, CURRENT_DIR, MODEL_DIR
from Haar import Detector


class HaarStreamer:
    def __init__(self, stream, path_to_classifier=os.path.join(CURRENT_DIR, MODEL_DIR, 'cascade.xml')):
        self.stream = stream
        self.detector = Detector(path_to_classifier)

    def process_stream(self, res_dir=''):
        cap = cv2.VideoCapture(self.stream)

        ret, frame = cap.read()
        out_stream = cv2.VideoWriter(res_dir, -1, 20, frame.shape[::-1][1::])
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = self.detector.detect_image(frame, minNeighbors=70, minSize=(90, 100))
            if len(res_dir):
                out_stream.write(frame)
            else:
                cv2.imshow('Recognition', res)
            sleep(0.02)

            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        out_stream.release()
        cv2.destroyAllWindows()
