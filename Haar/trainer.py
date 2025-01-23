import subprocess
from environments import CV_CREATESAMPLES_PATH, CV_TRAINCASCADE_PATH


class Trainer:

    @classmethod
    def create_vec_file(cls, good_file_path, vec_file_path, w, h):
        subprocess.run(f'{CV_CREATESAMPLES_PATH} -info {good_file_path} -vec {vec_file_path} -w {w} -h {h}', shell=True)

    @classmethod
    def train_cascade(cls, data_dir, vec_file_path, bad_file_path, numPos, numNeg, numStages, w, h):
        subprocess.run(f'{CV_TRAINCASCADE_PATH} -data {data_dir} -vec {vec_file_path} -bg {bad_file_path} -numPos {numPos} -numNeg {numNeg} -numStages {numStages}, -w {w} -h {h} -mode ALL', shell=True)
