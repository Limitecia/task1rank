import argparse
from collections import Counter, defaultdict

import utils
from src.service_codalab import CodalabSemeval2024T3Service, find_emotions_in_annotation
from src.conversation import iter_pairs
from src.service_csv import CsvService


def iter_cause_matrix_google_spreadsheet_fmt(cause_ctr, do_norm, cols, sep="\t"):

    def _hv(item, total):
        assert(isinstance(item, int))
        if total == 0:
            return "nan"
        return round(item / total * 1.0, 2) if do_norm else item

    # Print header.
    yield sep.join([label for label in [" "] + cols])
    for l in cols:
        v = cause_ctr[l]
        total = sum(v.values())
        col_name = [l]
        content = [str(_hv(v[ll], total)) for ll in cols]
        yield sep.join(col_name + content)


def print_counter_as_spreadsheet_fmt(ctr, cols, sep="\t"):
    return sep.join([str(ctr[col_name]) for col_name in cols])


def print_submission_stat(cfg, conversations):
    assert(cfg["s_type"] in ["etalon", "predict"])

    cause_annot = Counter()
    e_counter = Counter()
    cause_s = defaultdict(Counter)
    cause_t = defaultdict(Counter)
    text_spans = []
    text_original = []
    emotions_per_diff = defaultdict(list)
    pairs_it = iter_pairs(json_data=conversations,
                          max_dist=None,
                          return_annotation=True,
                          self_distant_relations=True)

    for c_id, c, u1, u2, a in pairs_it:

        diff = u2["utterance_ID"] - u1["utterance_ID"]

        r = find_emotions_in_annotation(a, src_utt_id=u1["utterance_ID"], tgt_utt_id=u2["utterance_ID"])

        if r is None:
            continue

        emotions_per_diff[diff].append(len(r))

        e, e_text = r[0]

        cause_s[u1["emotion"] if "emotion" in u1 else "N/A"][e] += 1
        cause_t[u2["emotion"] if "emotion" in u2 else "N/A"][e] += 1
        cause_annot[diff] += 1
        e_counter[e] += 1

        if cfg["s_type"] == "predict":
            begin, end = CodalabSemeval2024T3Service.annotation_parse_span(e_text)
            cause_text = CodalabSemeval2024T3Service.span_to_utterance(utterance_text=u1["text"], begin=begin, end=end)
            text_spans.append([cause_text])
            text_original.append([u1["text"]])

    # Columns setup.
    dist_cols = list(sorted(cause_annot.keys()))
    emotion_cols = ["joy", "surprise", "anger", "sadness", "disgust", "fear", "neutral"]

    total_causes = sum(cause_annot.values())

    print("Causes Distances Stat:")
    print("\t".join([str(itm) for itm in dist_cols]))
    print(print_counter_as_spreadsheet_fmt(ctr=cause_annot, cols=dist_cols))

    print("Causes Emotions Stat:")
    print("\t".join(emotion_cols))
    print(print_counter_as_spreadsheet_fmt(ctr=e_counter, cols=emotion_cols))

    print(f"Causes Total: {total_causes}")

    print(f"Conversations Total: {len(conversations)}")

    print(f"Avg. per conversation (total): {round(total_causes / len(conversations), 2)}")

    print(f"Avg. per distance:")
    ctr = Counter({d: round(v / len(conversations), 2) for d, v in cause_annot.most_common()})
    print("\t".join([str(itm) for itm in dist_cols]))
    print(print_counter_as_spreadsheet_fmt(ctr=ctr, cols=dist_cols))

    if cfg["print_covering_stat"]:
        ctr = Counter()
        for top_k in range(1, len(cause_annot) + 1):
            covered = sum([v for _, v in cause_annot.most_common(top_k)])
            ctr[top_k] = 100*round(covered/total_causes, 3)
        print(print_counter_as_spreadsheet_fmt(ctr=ctr, cols=[c + 1 for c in dist_cols]))

    print('-----')
    print(f"Character emotions states and their cause")
    print('-----')
    line_it = iter_cause_matrix_google_spreadsheet_fmt(cause_ctr=cause_s, do_norm=True, cols=emotion_cols)
    for line in line_it:
        print(line)
    print('-----')
    print(f"Target character state and emotions causes to them")
    print('-----')
    line_it = iter_cause_matrix_google_spreadsheet_fmt(
        cause_ctr=cause_t, do_norm=True,
        cols=["joy", "surprise", "anger", "sadness", "disgust", "fear", "neutral"])
    for line in line_it:
        print(line)

    if 'neutral' in e_counter:
        raise Exception("File should not contain neutral causes!")
    for s in [cause_s, cause_t]:
        for state in s.keys():
            for k in s[state].keys():
                if k == "neutral":
                    raise Exception("Person can't cause neutral emotion")

    if cfg["s_type"] == "predict":
        CsvService.write(target=cfg["causes_texts_csv"], lines_it=iter(text_spans), header=["causes_text"])
        CsvService.write(target=cfg["causes_texts_csv"]+".origin.csv", lines_it=iter(text_original), header=["causes_text"])

    # Emotions per diff
    print('-----')
    print("Emotions per diff")
    print('-----')
    for d, stat in emotions_per_diff.items():
        avg = sum(stat) / len(stat)
        print(f"Distance {d}: {round(avg, 2)}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src', type=str, default=utils.TRAIN_SRC)
    args = parser.parse_args()

    print(f"File Source: {args.src}")
    print_submission_stat(conversations=CodalabSemeval2024T3Service.read_data(args.src),
                          cfg={"s_type": "etalon", "print_covering_stat": True})
