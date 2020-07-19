from environs import Env
import os
from os.path import join, exists

env = Env()
env.read_env("MaLSTM.conf", recurse=False)

# model configuration
OUTPUT_UNITS = env.int("OUTPUT_UNITS")
MAXLEN = env.int("MAXLEN")
NO_OF_UNIQUE_WORDS = env.int("NO_OF_UNIQUE_WORDS")
EMBED_DIM = env.int("EMBED_DIM")
HOLDOUT_RATIO = env.float("HOLDOUT_RATIO")
BATCH_SIZE = env.int("BATCH_SIZE")
LEARNING_RATE = env.float("LEARNING_RATE")
NO_OF_ITERATIONS = env.int("NO_OF_ITERATIONS")
SEED = env.int("SEED")
FOLDER = env.str("ROOT_LOGDIR")

if not exists(join(os.getcwd(), FOLDER)):
    os.mkdir(join(os.getcwd(), FOLDER))
ROOT_LOGDIR = join(os.getcwd(), FOLDER)


