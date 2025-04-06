import argparse
import json
from os.path import join
from zipfile import ZipFile

from task_statistics_json import print_submission_stat
from utils import DATA_DIR


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src', type=str,
                        default=join(DATA_DIR, "e3_pair_ft/paper/submissions/3-flan-t5-base-thor-rr-spanfix-final.zip"))
    args = parser.parse_args()

    with ZipFile(args.src, 'r') as archive:
        fname = archive.namelist()[0]
        conversations = json.load(archive.open(fname))

    print(f"File Source: {args.src}")
    print_submission_stat(conversations=conversations, cfg={"s_type": "etalon", "print_covering_stat": True})
