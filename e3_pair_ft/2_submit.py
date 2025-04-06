import argparse
from collections import Counter
from os.path import join

import utils
from e3_pair_ft.utils_e import EXP_DIR, TASK_CLASSES_WITH_NEUTRAL, vocabulary_based_handler, read_entry_vocab
from src.service_codalab import fill_answers, CodalabSemeval2024T3Service
from src.service_csv import CsvService
from task_stat import print_submission_stat

meta_cols = ["c_id", "u1", "u2"]


def iter_answers_cause_and_state(labels_csv):
    for d in CsvService.read(target=labels_csv, skip_header=True, cols=meta_cols + ["cause", "state"]):
        predict_emotion_cause = TASK_CLASSES_WITH_NEUTRAL[int(d[-2])]
        predict_emotion_state = TASK_CLASSES_WITH_NEUTRAL[int(d[-1])]
        yield d[:len(meta_cols)] + [predict_emotion_cause, predict_emotion_state] + [None]


def iter_answers_cause(labels_csv):
    for d in CsvService.read(target=labels_csv, skip_header=True, cols=meta_cols + ["cause"]):
        predict_emotion_cause = TASK_CLASSES_WITH_NEUTRAL[int(d[-1])]
        yield d[:len(meta_cols)] + [predict_emotion_cause, None] + [None]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Infer Instruct LLM inference")
    parser.add_argument('--src', dest='src', type=str, default=utils.TEST_SRC)
    parser.add_argument('--src-labels', dest='src_labels', type=str, default=join(EXP_DIR, "mult", "google_flan-t5-base-thor_cause_rr-30-01.csv.meta.csv"))
    parser.add_argument('--labels-type', dest='labels_type', type=str, default="single", choices=["single", "joined"])

    args = parser.parse_args()

    # Providing vocabulary detail for spans correction.
    span_log_c = Counter()
    c_p_stat = Counter()
    c_s_stat = Counter()
    prefix_vocab = read_entry_vocab(vocab_csv_filepath=join(EXP_DIR, "vocab_prefix.csv"))
    suffix_vocab = read_entry_vocab(vocab_csv_filepath=join(EXP_DIR, "vocab_suffix.csv"))

    answers_iter_funcs = {
        "single": iter_answers_cause,
        "joined": iter_answers_cause_and_state
    }

    src_data = CodalabSemeval2024T3Service.read_data(args.src)
    fill_answers(src_data=src_data,
                 answers_it=answers_iter_funcs[args.labels_type](labels_csv=args.src_labels),
                 handle_span=lambda text, b, e: vocabulary_based_handler(
                     utterance_text=text, span_begin=b, span_end=e,
                     prefix_vocab=prefix_vocab, suffix_vocab=suffix_vocab,
                     c_log=span_log_c,
                     c_p_stat=c_p_stat,
                     c_s_stat=c_s_stat))

    # Save.
    target = join(EXP_DIR, f"submission-{args.labels_type}.zip")
    CodalabSemeval2024T3Service.save_submission(target=target, st1_json_data=src_data)
    # Calculate state.
    print_submission_stat(conversations=src_data,
                          cfg={"s_type": "predict",
                               "print_covering_stat": False,
                               "causes_texts_csv": f"{target}.paths.csv"})
    print("Spans Correction Stat:", span_log_c)
    print("Prefixes removed Stat:", c_p_stat)
    print("Suffixes removed Stat:", c_s_stat)
