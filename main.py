from Haar import DataPreparator
import os
from environments import BOO_DIR, CART_DIR, CURRENT_DIR


if __name__ == '__main__':
    # todo: choice of Haar vs YOLO with arguments from cmd

    dp = DataPreparator(os.path.join(CURRENT_DIR, BOO_DIR))
    dp.prepare_labels()
