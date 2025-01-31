import os.path
from time import sleep
import cv2
from environments import CURRENT_DIR, WEIGHTS_DIR
from YOLO import YoloDetector


cap = cv2.VideoCapture(os.path.join(CURRENT_DIR, 'mk_boo.mp4'))
detector = YoloDetector(os.path.join(WEIGHTS_DIR, 'best.pt'))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = detector.detect_image(frame)
    cv2.imshow('Recognition', res)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
