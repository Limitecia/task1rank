import json
import string
from zipfile import ZipFile

from src.conversation import try_find_conversation, try_find_utterance


class CodalabSemeval2024T3Service(object):

    answers_key = "emotion-cause_pairs"

    NEUTRAL_EMOTION_LOWER = "neutral"
    TASK_CLASSES = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
    TASK_CLASSES_LOWER = [c.lower() for c in TASK_CLASSES]

    @staticmethod
    def read_data(src):
        assert(isinstance(src, str))
        with open(src, "r") as f:
            return json.load(f)

    @staticmethod
    def annotation_parse_span(span_text):
        """ 0_14 -> text
        """
        span_begin, span_end = [int(s) for s in span_text.split("_")]
        return span_begin, span_end

    @staticmethod
    def span_to_utterance(utterance_text, begin, end):
        assert(isinstance(utterance_text, str))
        terms = utterance_text.split()
        assert(0 <= end <= len(terms))
        assert(0 <= begin <= len(terms))
        return " ".join(terms[begin:end])

    @staticmethod
    def annotation_parse_utterance_id(utt):
        """ '1_I realize I am totally naked .' -> 1.
        """
        assert (isinstance(utt, str))
        content = utt.split('_')
        utt_id = content[0]
        text = "_".join(content[1:])
        assert (isinstance(text, str))
        return int(utt_id), text

    @staticmethod
    def annotation_parse_emotion(e):
        """ '4_joy` -> `joy`
        """
        args = e.split('_')
        assert (len(args) == 2)
        return [int(args[0]), args[1]]

    @staticmethod
    def normalize_utterance(text):
        """ An attempt to perform utterance text normalization, similar to the ECAC-2024 competitions format.
        """

        text = text.replace("...", "[CDOT]")

        for sign in string.punctuation:
            if sign in "`'[]":
                continue
            text = text.replace(sign, " {} ".format(sign))

        text = text.replace("[CDOT]", "...")

        return " ".join(text.split())

    @staticmethod
    def is_term_punctuation(term):
        assert(isinstance(term, str))

        is_all_punkt_chars = True
        for char in term:
            if char not in string.punctuation:
                is_all_punkt_chars = False

        return is_all_punkt_chars

    @staticmethod
    def utterance_to_terms(utterance):
        return utterance.strip().split()

    @staticmethod
    def default_utterance_span(utterance):
        terms = CodalabSemeval2024T3Service.utterance_to_terms(utterance)
        return 0, len(terms)

    @staticmethod
    def default_correct_utterance_span(utterance, begin=None, end=None):
        """ Please note that, for accurate evaluation, your cause span
            should not include the punctuation token at the beginning and end
            source: https://codalab.lisn.upsaclay.fr/competitions/16141#learn_the_details-submission-format
        """
        assert(isinstance(utterance, str))

        terms = CodalabSemeval2024T3Service.utterance_to_terms(utterance)
        default_begin, default_end = CodalabSemeval2024T3Service.default_utterance_span(utterance)

        # Default span region.
        begin = default_begin if begin is None else begin
        end = default_end if end is None else end

        assert(begin >= 0 and end <= len(terms))

        # Apply span regions.
        terms = terms[begin:end]

        while (len(terms) > 0) and (CodalabSemeval2024T3Service.is_term_punctuation(terms[0])):
            terms = terms[1:]
            begin += 1

        while (len(terms) > 0) and (CodalabSemeval2024T3Service.is_term_punctuation(terms[-1])):
            terms = terms[:-1]
            end -= 1

        assert(begin != end)

        return (begin, end) if end - begin > 0 else (default_begin, default_end)

    @staticmethod
    def fix_submission(json_data):
        """ Guarantee that answers, including empty one, were provided for each conversation
        """
        for conversation in json_data:
            if CodalabSemeval2024T3Service.answers_key not in conversation:
                conversation[CodalabSemeval2024T3Service.answers_key] = []

    @staticmethod
    def check_submission(json_data):
        for conversation in json_data:
            # Confirm that every entry has answers key.
            assert(CodalabSemeval2024T3Service.answers_key in conversation)
            answers = ["-".join(a) for a in conversation[CodalabSemeval2024T3Service.answers_key]]
            # Confirm that we do not duplicate entries.
            assert(len(set(answers)) == len(answers))

    @staticmethod
    def save_submission(target, st1_json_data=None, st2_json_data=None, notify=True):
        with ZipFile(target, "w") as zip_file:
            for task_ind, json_data in enumerate([st1_json_data, st2_json_data]):
                if json_data is None:
                    continue
                CodalabSemeval2024T3Service.fix_submission(json_data)
                CodalabSemeval2024T3Service.check_submission(json_data)
                json_bytes = json.dumps(json_data, ensure_ascii=False, indent=4)
                zip_file.writestr(f'Subtask_{task_ind+1}_pred.json', json_bytes)

        if notify:
            print(f"Saved: {target}")


def find_emotions_in_annotation(annotation, src_utt_id, tgt_utt_id):
    assert(isinstance(annotation, list))

    r = []
    for cause in annotation:
        annot_caused_emotion, annot_src_utt = cause
        annot_src_utt_id, text = CodalabSemeval2024T3Service.annotation_parse_utterance_id(annot_src_utt)
        annot_tgt_utt, emotion = CodalabSemeval2024T3Service.annotation_parse_emotion(annot_caused_emotion)
        if annot_src_utt_id == src_utt_id and annot_tgt_utt == tgt_utt_id:
            r.append([emotion, text])

    # No results.
    if len(r) == 0:
        return None

    return r


def find_emotion_in_annotation(annotation, src_utt_id, tgt_utt_id, default=None):
    assert(isinstance(annotation, list))
    r = find_emotions_in_annotation(annotation=annotation, src_utt_id=src_utt_id, tgt_utt_id=tgt_utt_id)
    return [default, None] if r is None else r[0]


def fill_answers(src_data, answers_it, handle_span=None):
    """ This is the main script that allow to form a submission data.
        answers_it: iter
            each row represent a tuple/list of (conv_id, utt1_id, utt2_id, predict_class, spans)
            spans are optional.
    """
    answers = {}

    for conv_id, utt1_id, utt2_id, predict_emotion_cause, predict_emotion_state, spans in answers_it:
        assert((predict_emotion_cause in CodalabSemeval2024T3Service.TASK_CLASSES_LOWER)
               or predict_emotion_cause == CodalabSemeval2024T3Service.NEUTRAL_EMOTION_LOWER
               or predict_emotion_cause is None)
        assert((predict_emotion_state in CodalabSemeval2024T3Service.TASK_CLASSES_LOWER)
               or predict_emotion_state == CodalabSemeval2024T3Service.NEUTRAL_EMOTION_LOWER
               or predict_emotion_state is None)

        if conv_id not in answers:
            answers[conv_id] = []

        # Registering the composed answer.
        u = try_find_utterance(json_data=src_data, conv_id=int(conv_id), utt_id=int(utt1_id))
        text = u["text"]

        # Registering emotion state.
        if predict_emotion_state is not None:
            u["emotion"] = predict_emotion_state

        if predict_emotion_cause is None:
            continue

        if predict_emotion_cause == CodalabSemeval2024T3Service.NEUTRAL_EMOTION_LOWER:
            continue

        # Extract spans optionally.
        if spans is None:
            begin = None
            end = None
        else:
            begin, end = spans

        # Span handling.
        if handle_span is not None:
            span_begin, span_end = CodalabSemeval2024T3Service.default_utterance_span(text)
            span_begin, span_end = handle_span(text, span_begin, span_end)
            span_begin, span_end = CodalabSemeval2024T3Service.default_correct_utterance_span(
                text, begin=span_begin, end=span_end)
        else:
            span_begin, span_end = CodalabSemeval2024T3Service.default_correct_utterance_span(
                text, begin=begin, end=end)

        answers[conv_id].append([f"{utt2_id}_{predict_emotion_cause}".lower(), f"{utt1_id}_{span_begin}_{span_end}"])

    # Provide the answers for into the conversation data.
    for conv_id, ans in answers.items():
        conversation = try_find_conversation(json_data=src_data, conv_id=int(conv_id))
        conversation[CodalabSemeval2024T3Service.answers_key] = ans
