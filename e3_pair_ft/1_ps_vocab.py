import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
from collections import Counter
from os.path import join

from utils_e import EXP_DIR
from src.service_codalab import find_emotion_in_annotation, CodalabSemeval2024T3Service
from src.conversation import iter_pairs
from src.service_csv import CsvService


from utils import TRAIN_SRC


def extract_ps(text_u, text_c):
    to = text_u.index(text_c)
    prefix = text_u[:to]
    suffix = text_u[to + len(text_c):]
    return prefix, suffix


def keep_entry(entry, v_type, max_len=3):
    assert(isinstance(entry, str))
    assert(v_type in ["prefix", "suffix"])

    terms = entry.split()
    if len(terms) > max_len or len(terms) == 0:
        return False

    if v_type == "suffix":
        if not CodalabSemeval2024T3Service.is_term_punctuation(terms[0]):
            return False
        return True

    if v_type == "prefix":
        if not CodalabSemeval2024T3Service.is_term_punctuation(terms[-1]):
            return False
        return True


def compose_vocabularies(prefix_count, suffix_count):
    """ This code is related to manual vocabulary creation, utilized
        for pattern-matching method of preventing prefixes and suffixes
    """

    for v in list(prefix_count.keys()):
        if not keep_entry(v, v_type="prefix"):
            del prefix_count[v]

    for v in list(suffix_count.keys()):
        if not keep_entry(v, v_type="suffix"):
            del suffix_count[v]

    for d_type, data in {"prefix": prefix_count.most_common(), "suffix": suffix_count.most_common()}.items():
        CsvService.write(target=join(EXP_DIR, f"vocab_{d_type}.csv"),
                         lines_it=[(v.strip(), count) for v, count in data],
                         header=["entries", "count"],
                         force_mk_dir=True)


def handle_conversations(conversations):
    pc = Counter()
    sc = Counter()
    pairs_it = iter_pairs(json_data=conversations, max_dist=None, return_annotation=True, self_distant_relations=True)
    for c_id, c, u1, u2, a in pairs_it:
        e, e_text = find_emotion_in_annotation(a, src_utt_id=u1["utterance_ID"], tgt_utt_id=u2["utterance_ID"])
        if e_text is None:
            continue
        prefix, suffix = extract_ps(text_u=u1["text"], text_c=e_text)
        pc[prefix] += 1
        sc[suffix] += 1

    return pc, sc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src', type=str, default=TRAIN_SRC)
    parser.add_argument('--output', dest='output', type=str, default=None)
    args = parser.parse_args()

    print(f"File Source: {args.src}")
    prefix_count, suffix_count = handle_conversations(conversations=CodalabSemeval2024T3Service.read_data(args.src))
    compose_vocabularies(prefix_count=prefix_count, suffix_count=suffix_count)
