import os.path
import time

import cv2
from environments import MK_GAMEPLAY, SCREEN_TIME, GP_DIR, CURRENT_DIR

cap = cv2.VideoCapture(MK_GAMEPLAY)

t0 = int(time.time())
i=1

gameplay_dir = os.path.join(CURRENT_DIR, GP_DIR)
if not os.path.exists(gameplay_dir):
    os.makedirs(gameplay_dir)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('vid', frame)
    if time.time() - t0 >= SCREEN_TIME:
        cv2.imwrite(os.path.join(gameplay_dir, str(i*SCREEN_TIME) + '.jpg'), frame)
        t0 = time.time()
        i+=1

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
