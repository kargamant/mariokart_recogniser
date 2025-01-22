from Haar import DataPreparator
import os
from environments import BOO_DIR, KART_DIR, CURRENT_DIR, GP_DIR


if __name__ == '__main__':
    # todo: choice of Haar vs YOLO with arguments from cmd
    # todo: also make an option to choose between boo and coins recognition

    dp = DataPreparator(os.path.join(CURRENT_DIR, BOO_DIR), os.path.join(CURRENT_DIR, GP_DIR))
    dp.prepare_labels()
