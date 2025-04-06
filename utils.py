from os.path import dirname, realpath, join

current_dir = dirname(realpath(__file__))
DATA_DIR = join(current_dir, "data")

TRAIN_SRC = join(DATA_DIR, "Subtask_1_train.json")
TRIAL_SRC = join(DATA_DIR, "Subtask_1_trial.json")
TEST_SRC = join(DATA_DIR, "Subtask_1_test.json")

DEFAULT_SRC = TRIAL_SRC
