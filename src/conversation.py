import json


def try_find_conversation(json_data, conv_id):
    assert(isinstance(json_data, list))
    assert(isinstance(conv_id, int))
    for conversation in json_data:
        if conversation["conversation_ID"] == conv_id:
            return conversation


def try_find_utterance(json_data, conv_id, utt_id):
    assert(isinstance(utt_id, int))

    related_conversation = try_find_conversation(json_data, conv_id)
    if related_conversation is None:
        return None

    for utterance in related_conversation["conversation"]:
        if utterance["utterance_ID"] == utt_id:
            return utterance


def iter_conversations(json_data, return_annotation=False):
    assert(isinstance(json_data, list))
    for conversation in json_data:
        cid = conversation["conversation_ID"]
        annotation = conversation["emotion-cause_pairs"] if "emotion-cause_pairs" in conversation else []
        yield [cid, conversation["conversation"]] + ([annotation] if return_annotation else [])


def iter_pairs(json_data, max_dist=None, return_annotation=False, self_distant_relations=False):
    for data in iter_conversations(json_data, return_annotation=return_annotation):
        cid = data[0]
        conversation = data[1]
        annotation = data[2] if return_annotation else None
        for utt_src in conversation:
            for utt_tgt in conversation:

                utt_src_id = utt_src["utterance_ID"]
                utt_tgt_id = utt_tgt["utterance_ID"]

                # Consider only relations to the front.
                if utt_src_id > utt_tgt_id:
                    continue

                # We cannot affect ourself.
                if not self_distant_relations and (utt_src["speaker"] == utt_tgt["speaker"] and utt_src_id != utt_tgt_id):
                    continue

                # Filter by max distance.
                if max_dist is not None and utt_tgt_id - utt_src_id > max_dist:
                    continue

                yield [cid, conversation, utt_src, utt_tgt] + ([annotation] if annotation is not None else [])


def extract_conversation_context(conversation, utt, window, ust_prefix):
    assert(isinstance(conversation, list))
    assert(isinstance(utt, dict))
    assert(isinstance(window, int))

    u_taken = []
    for u in conversation:
        if u["utterance_ID"] > utt["utterance_ID"]:
            continue
        if abs(u["utterance_ID"] - utt["utterance_ID"]) > window:
            continue
        u_taken.append(u)

    return [ust_prefix.format(uid=u["utterance_ID"], speaker=u["speaker"], text=u["text"]) for u in u_taken]
