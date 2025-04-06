import argparse
from collections import Counter
from os.path import join

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils

from e3_pair_ft.utils_e import EXP_DIR, MAX_DIALOGUE_HISTORY, UST_PREFIX, train_test_split_content
from src.service_codalab import CodalabSemeval2024T3Service
from src.conversation import iter_conversations, extract_conversation_context
from src.service_csv import CsvService


def iter_contexts(json_data, window):
    row_id = 0
    counter = Counter()
    for cid, conversation in iter_conversations(json_data):
        for utt in conversation:
            context = extract_conversation_context(conversation=conversation, utt=utt, window=window,
                                                   ust_prefix=UST_PREFIX)
            yield [row_id, cid, utt["utterance_ID"], " ".join(context), utt["text"], utt["emotion"]]
            counter[utt["emotion"]] += 1
            row_id += 1
    print(counter)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Infer Instruct LLM Inference for Emotion Prediction")
    parser.add_argument('--source', dest='source', type=str, default=utils.TRAIN_SRC)
    parser.add_argument('--window_history', dest='window_history', type=int, default=MAX_DIALOGUE_HISTORY)
    parser.add_argument('--train_part', dest='train_part', type=float, default=0.9)
    parser.add_argument('--out-dir', dest='out_dir', type=str, default=EXP_DIR)

    args = parser.parse_args()

    train_data, valid_data = train_test_split_content(data=CodalabSemeval2024T3Service.read_data(args.source),
                                                      proportion=args.train_part)

    for d_type, json_data in {"train": train_data, "valid": valid_data}.items():
        contexts_iter = iter_contexts(json_data=json_data, window=args.window_history)
        target = join(args.out_dir, f"state-mult-{d_type}.csv")
        CsvService.write(target=target, lines_it=contexts_iter,
                         header=["row_id", "conversation_id", "utterance_id", "context", "target", "emotion"])
