import argparse
from os.path import join

import utils
from src.service_codalab import find_emotion_in_annotation, CodalabSemeval2024T3Service
from src.conversation import iter_pairs
from src.service_csv import CsvService


def iter_content(conversations):
    pairs_it = iter_pairs(json_data=conversations, max_dist=None,
                          return_annotation=True, self_distant_relations=True)
    yield ["c_id", "u1", "u2", "cause"]
    for c_id, c, u1, u2, a in pairs_it:
        e, e_text = find_emotion_in_annotation(a, src_utt_id=u1["utterance_ID"], tgt_utt_id=u2["utterance_ID"])
        if e is None:
            continue
        e_index = CodalabSemeval2024T3Service.TASK_CLASSES_LOWER.index(e)
        yield c_id, u1["utterance_ID"], u2["utterance_ID"], e_index


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Infer Instruct LLM inference")
    parser.add_argument('--src', dest='src', type=str, default=utils.TRAIN_SRC)
    parser.add_argument('--out', dest='out', type=str, default="out.txt")
    args = parser.parse_args()

    json_conversation = CodalabSemeval2024T3Service.read_data(args.src)
    CsvService.write(target=args.out, lines_it=iter_content(json_conversation))