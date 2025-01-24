import os.path
from time import sleep
import cv2
from environments import MK_GAMEPLAY, SCREEN_TIME, GP_DIR, CURRENT_DIR, MODEL_DIR
from Haar import Detector


cap = cv2.VideoCapture(os.path.join(CURRENT_DIR, 'mk_boo.mp4'))
detector = Detector(os.path.join(CURRENT_DIR, MODEL_DIR, 'cascade.xml'))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    new_frame = detector.detect_image(frame, minNeighbors=70, minSize=(90, 100))
    cv2.imshow('Recognition', new_frame)
    sleep(0.05)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
