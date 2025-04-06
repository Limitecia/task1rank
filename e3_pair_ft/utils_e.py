from collections import Counter
from os.path import join

from src.service_codalab import CodalabSemeval2024T3Service, find_emotion_in_annotation
from src.conversation import extract_conversation_context, iter_pairs
from src.service_csv import CsvService
from utils import DATA_DIR


MAX_CAUSE_DISTANCE = 3
MAX_DIALOGUE_HISTORY = 3
UST_PREFIX = "> {speaker}: `{text}`"

TASK_CLASSES_WITH_NEUTRAL = CodalabSemeval2024T3Service.TASK_CLASSES_LOWER + \
                            [CodalabSemeval2024T3Service.NEUTRAL_EMOTION_LOWER]

EXP_DIR = join(DATA_DIR, "e3_pair_ft")


def iter_conversation_pairs(cfg, json_data, keep_annotation=True):
    pairs_it = iter_pairs(json_data=json_data, max_dist=cfg["utt_dist"], return_annotation=keep_annotation,
                          self_distant_relations=cfg["self_distant_relations"])
    for pairs_data in pairs_it:
        c_id, c, u1, u2 = pairs_data[:4]
        u1_id = u1["utterance_ID"]
        u2_id = u2["utterance_ID"]
        meta = [c_id, u1_id, u2_id]
        if keep_annotation:
            e_cause, _ = find_emotion_in_annotation(
                pairs_data[4], src_utt_id=u1_id, tgt_utt_id=u2_id,
                default=CodalabSemeval2024T3Service.NEUTRAL_EMOTION_LOWER)
        else:
            e_cause = CodalabSemeval2024T3Service.NEUTRAL_EMOTION_LOWER
        conversation_context = extract_conversation_context(
            conversation=c, utt=u2, window=cfg["h_size"], ust_prefix=UST_PREFIX)
        yield [" ".join(conversation_context), u1, e_cause], meta


def iter_dataset(data_it, oversampling_it_func, data_handle_func):
    assert(callable(data_handle_func))
    ctr = Counter()
    for data, meta in data_it:
        for m_data in oversampling_it_func(data):
            ctr[m_data[-1]] += 1               # Emotion cause.
            m_data = data_handle_func(m_data) if data_handle_func is not None else m_data
            yield m_data + meta
    print(ctr)


def train_test_split_content(data, proportion):
    assert(isinstance(proportion, float))
    border = int(len(data) * proportion)
    return data[:border], data[border:]


def vocabulary_based_handler(utterance_text, span_begin, span_end, prefix_vocab, suffix_vocab,
                             c_log, c_p_stat, c_s_stat):
    """ This is a vocabulary-based method of utterances span correction.
    """
    assert(isinstance(prefix_vocab, list))
    assert(isinstance(suffix_vocab, list))

    def __calc_offset(text):
        return len(text.split())

    cause_text = CodalabSemeval2024T3Service.span_to_utterance(
        utterance_text=utterance_text, begin=span_begin, end=span_end)
    cause_text = cause_text.lower()

    new_span_begin = span_begin
    new_span_end = span_end
    updated = False
    while True:
        local_update = False
        for prefix in prefix_vocab:
            prefix = prefix.lower()
            if cause_text.startswith(prefix):
                new_span_begin += __calc_offset(prefix)
                cause_text = cause_text[len(prefix):].strip()
                c_log["p"] += 1
                c_p_stat[prefix] += 1
                updated = True
                local_update = True
                break
        if local_update is False:
            break
    while True:
        local_update = False
        for suffix in suffix_vocab:
            suffix = suffix.lower()
            if cause_text.endswith(suffix):
                new_span_end -= __calc_offset(suffix)
                cause_text = cause_text[:-len(suffix)].strip()
                c_s_stat[suffix] += 1
                updated = True
                local_update = True
                break
        if local_update is False:
            break

    c_log["fixed" if updated else "as-it-is"] += 1

    # Check that the new borders are correct.
    if new_span_begin < new_span_end:
        return new_span_begin, new_span_end
    else:
        c_log["reverted"] += 1
        return span_begin, span_end


def read_entry_vocab(vocab_csv_filepath, top_c=None):
    # NOTE: we consider that the vocabulary is in descended order of relevant chars
    vocab = [l[0] for l in CsvService.read(target=vocab_csv_filepath, skip_header=True, cols=["entries"])]
    vocab = vocab[:top_c] if top_c is not None else vocab
    vocab = sorted(vocab, key=lambda item: len(item), reverse=True)
    return vocab
