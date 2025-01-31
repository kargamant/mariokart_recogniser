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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            new_frame = self.detector.detect_image(frame, minNeighbors=70, minSize=(90, 100))
            cv2.imshow('Recognition', new_frame)
            sleep(0.02)

            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
