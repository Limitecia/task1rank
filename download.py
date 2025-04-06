import os
from os.path import join

from src.utils import download
from utils import DATA_DIR


data = {
    # Original data for training stage.
    join(DATA_DIR, "Subtask_1_train.json"):
        "https://www.dropbox.com/scl/fi/jec8w0v76qgpit95smako/Subtask_1_train.json?rlkey=pzgr6gjb5ufqd2mlzhtvq7n76&dl=1",
    # Original data for the trial stage.
    join(DATA_DIR, "Subtask_1_trial.json"):
        "https://www.dropbox.com/scl/fi/yio48tkxcxpxmuym9ckk4/Subtask_1_trial.json?rlkey=wogzwdx1fka2cdq1asu4zdo0d&dl=1",
    join(DATA_DIR, "Subtask_1_test.json"):
        "https://www.dropbox.com/scl/fi/7ezhdruesjj3z2waf73hy/Subtask_1_test.json?rlkey=gjja37s5q0b9flul6hxkqnyi5&dl=1",
}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    

for target, url in data.items():
    download(dest_file_path=target, source_url=url)
