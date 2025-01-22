import os
import cv2


class DataPreparator:
    def __init__(self, dir: str):
        self.directory = dir

    def prepare_labels(self):
        train_dir = os.path.join(self.directory, "train")
        labels = os.path.join(train_dir, "labels")
        images = os.path.join(train_dir, "images")


        with open(os.path.join(self.directory, "Good.dat"), 'w+') as good_file:
            for label in os.listdir(labels):
                image = os.path.join(images, label.replace('.txt', '.jpg'))
                with open(os.path.join(labels, label), "r") as file:
                    im = cv2.imread(image)

                    to_write = self.to_haar(*list(map(float, file.readline().split(' '))), *im.shape)
                    self.drop_label_record(*to_write, good_file, image)


    def to_haar(self, class_label, centre_x_norm, centre_y_norm, width_norm, height_norm, h, w, c=3):
        class_label = int(class_label)

        # little size conversion
        centre_x = centre_x_norm * w
        centre_y = centre_y_norm * h
        width = int(width_norm * w)
        height = int(height_norm * h)

        # haar cascade format
        x1 = int(centre_x - width / 2)
        y1 = int(centre_y - height / 2)

        return class_label, x1, y1, width, height

    def drop_label_record(self, class_label, x1, y1, width, height, file, file_path):
        record = f'{file_path} {class_label} {x1} {y1} {width} {height}\n'
        file.write(record)

