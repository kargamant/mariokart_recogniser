from ultralytics import YOLO
from environments import YOLO_MODEL, YAML_DATA_DESCRIPTION

model = YOLO(YOLO_MODEL)

model.train(data=YAML_DATA_DESCRIPTION, epochs=7, rect=True)

model.save('boo_yolo.pt')
