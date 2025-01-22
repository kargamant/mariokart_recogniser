import os


BOO_DIR = 'boo/'
KART_DIR = 'coin_cart/'
GP_DIR = 'gameplay/'
MODEL_DIR = 'boo_detect/'
GOOD_FILE_DIR = 'Good.dat'
VEC_FILE_DIR = 'good.vec'
BAD_FILE_DIR = 'Bad.dat'
CURRENT_DIR = os.path.dirname(__file__)
MK_GAMEPLAY = os.path.join(CURRENT_DIR, "mk_gameplay.mp4")
SCREEN_TIME = 3 # screenshot every SCREEN_TIME seconds, needed for DataGathering module

# for subprocess
# I would really like to get it somewhere else, but it builds directory dependently
# For your specific case leave here path where you have built an opencv
CV_CREATESAMPLES_PATH = '/mnt/c/Users/Honor/Documents/cv_build/bin/opencv_createsamples'
CV_TRAINCASCADE_PATH = '/mnt/c/Users/Honor/Documents/cv_build/bin/opencv_traincascade'
