import utils.config_loader as config
from utils.config_loader import config_model
from data.dataset_parser import dataset_parser
import numpy as np


"""
    This module builds inputs for BERT-RR model.
    
    STATUS: *DONE*
"""

def _build_bert_tokens_for_sent(query_tokens, instance_tokens):
    in_size = [config_model['max_n_tokens'], ]

    token_ids = np.zeros(in_size, dtype=np.int32)
    seg_ids = np.zeros(in_size, dtype=np.int32)
    token_masks = np.zeros(in_size)

    # logger.info('shape of token_ids: {}'.format(token_ids.shape))
    if query_tokens:
        tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + instance_tokens + ['[SEP]']
        token_id_list = config.bert_tokenizer.convert_tokens_to_ids(tokens)
        n_tokens = len(token_id_list)
        seg_ids[len(query_tokens) + 2:n_tokens] = [1] * (len(instance_tokens) + 1)
    else:
        tokens = ['[CLS]'] + instance_tokens + ['[SEP]']
        token_id_list = config.bert_tokenizer.convert_tokens_to_ids(tokens)
        n_tokens = len(token_id_list)

    token_ids[:n_tokens] = token_id_list
    token_masks[:n_tokens] = [1] * n_tokens

    sent_in = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
    }

    return sent_in, tokens


def build_bert_sentence_x(query, sentence, with_sub, sub_token='[unused1]'):
    query_tokens = None
    if query:
        query_tokens = dataset_parser.parse_query(query)
        if with_sub:
            query_tokens = [sub_token if token == '[SUBQUERY]' else token for token in query_tokens]
            query_tokens = query_tokens[:config_model['max_nw_query']-1] + [sub_token]  # add at the end an extra token
    
    instance_tokens = dataset_parser.sent2words(sentence)[:config_model['max_nw_sent']]
    return _build_bert_tokens_for_sent(query_tokens, instance_tokens)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def build_bert_sentence_for_mn(query, sentence, with_sub, 
        sub_token='[unused1]', 
        cls_token='[CLS]',
        sep_token='[SEP]',
        pad_token=0,
        max_seq_length=256):
    """
        This function is for ranking sentences from MultiNews train/dev set.

    """
    query_tokens = None
    if query:
        query_tokens = dataset_parser.parse_query(query)
        
    instance_tokens = dataset_parser.sent2words(sentence)

    if with_sub: # CLS, SUBQUERY, SEP, SEP
        special_tokens_count = 4
    else:  # CLS, SEP, SEP
        special_tokens_count = 3

    _truncate_seq_pair(query_tokens, instance_tokens, max_seq_length-special_tokens_count)

    if with_sub:
        query_tokens = [sub_token if token == '[SUBQUERY]' else token for token in query_tokens]
        query_tokens += [sub_token]
    
    tokens = query_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    tokens += instance_tokens + [sep_token]
    segment_ids += [1] * (len(instance_tokens) + 1)

    tokens = [cls_token] + tokens
    segment_ids = [0] + segment_ids

    input_ids = config.bert_tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    token_ids = np.array(input_ids, dtype=np.int32)
    seg_ids = np.array(segment_ids, dtype=np.int32)
    token_masks = np.array(input_mask)

    sent_in = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
    }
    return sent_in, tokens
