import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils

import random
import argparse
from os.path import join
from e3_pair_ft.utils_e import iter_conversation_pairs, EXP_DIR, iter_dataset, \
    MAX_CAUSE_DISTANCE, MAX_DIALOGUE_HISTORY, train_test_split_content
from src.service_codalab import CodalabSemeval2024T3Service
from src.service_csv import CsvService


def no_label_sampling_mult(data):
    yield data


def handle_data_row(data):
    if len(data) == 2:
        print(data)
    c, u1, e_cause = data
    emotion_state = u1["emotion"] if "emotion" in u1 else CodalabSemeval2024T3Service.NEUTRAL_EMOTION_LOWER
    emotion_cause = e_cause
    return [c, u1["text"], emotion_state, emotion_cause]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src-train', dest='src_train', type=str, default=utils.TRAIN_SRC)
    parser.add_argument('--src-test', dest='src_test', type=str, default=utils.TEST_SRC)
    parser.add_argument('--pair-distance', dest='max_pair_dist', type=str, default=MAX_CAUSE_DISTANCE)
    parser.add_argument('--dialogue-history', dest='dialogue_history', type=str, default=MAX_DIALOGUE_HISTORY)
    parser.add_argument('--output-prompt', dest='output_prompt', type=str, default=join(EXP_DIR, f"cause-mult"))
    parser.add_argument('--train_part', dest='train_part', type=float, default=0.9)

    args = parser.parse_args()

    random.seed(42)

    cfg = {
        "utt_dist": args.max_pair_dist,
        "h_size": args.dialogue_history,
        "self_distant_relations": False
    }

    train, valid = train_test_split_content(
        data=CodalabSemeval2024T3Service.read_data(args.src_train),
        proportion=args.train_part)
    test = CodalabSemeval2024T3Service.read_data(src=args.src_test)

    for d_type, json_data in {"train": train, "valid": valid, "test": test}.items():
        keep_annotation = d_type != "test"
        d_it = iter_dataset(
            data_it=iter_conversation_pairs(cfg=cfg, json_data=json_data, keep_annotation=keep_annotation),
            data_handle_func=handle_data_row,
            oversampling_it_func=no_label_sampling_mult)
        infix = "" if MAX_CAUSE_DISTANCE == args.max_pair_dist else f"-sparse-d{args.max_pair_dist}"
        CsvService.write(target=f"{args.output_prompt}{infix}-{d_type}.csv", force_mk_dir=True, lines_it=d_it,
                         header=["context", "source", "emotion_state", "emotion_cause"] + ["c_id", "u1", "u2"])
