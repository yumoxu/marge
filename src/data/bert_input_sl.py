import utils.config_loader as config
from utils.config_loader import logger, config_model
import utils.tools as tools
from data.dataset_parser import dataset_parser
import numpy as np


def get_max_n_tokens(is_para):
    if is_para:
        return config_model['max_n_para_tokens']

    return config_model['max_n_query_tokens']


def _build_token_ids(words, max_n_tokens):
    token_ids = np.zeros([max_n_tokens, ], dtype=np.int32)
    # tokens = tools.flatten(sent_tokens)

    tokens = ['[CLS]'] + words
    token_id_list = config.bert_tokenizer.convert_tokens_to_ids(tokens)
    n_tokens = len(token_id_list)
    # logger.info('n_tokens: {}'.format(n_tokens))
    token_ids[:n_tokens] = token_id_list

    return token_ids, n_tokens


def _build_token_masks(n_tokens, max_n_tokens):
    token_masks = np.zeros([max_n_tokens, ])
    token_masks[:n_tokens] = [1] * n_tokens
    return token_masks


def _build_seg_ids(max_n_tokens):
    seg_ids = np.zeros([max_n_tokens, ], dtype=np.int32)
    return seg_ids


def _build_bert_tokens(words, max_n_tokens):
    token_ids, n_tokens = _build_token_ids(words, max_n_tokens)
    token_masks = _build_token_masks(n_tokens, max_n_tokens)
    seg_ids = _build_seg_ids(max_n_tokens)

    res = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
    }

    return res


def build_bert_x_doc_sl(doc_fp):
    # build para x
    doc_res = dataset_parser.parse_doc2sents(doc_fp)
    # init paras arrays
    max_ns_doc = config_model['max_ns_doc']
    max_nt = config_model['max_nt_sent']
    basic_doc_size = [max_ns_doc, max_nt]

    doc_token_ids = np.zeros(basic_doc_size, dtype=np.int32)
    doc_seg_ids = np.zeros(basic_doc_size, dtype=np.int32)
    doc_token_masks = np.zeros(basic_doc_size)

    # build para
    word_list = doc_res['words']
    # logger.info('word list from doc: {}'.format(word_list))
    for s_idx, words_s in enumerate(word_list):
        # logger.info('{}: {}'.format(s_idx, words_s))
        sent_bert_in = _build_bert_tokens(words=words_s, max_n_tokens=max_nt)
        doc_token_ids[s_idx] = sent_bert_in['token_ids']
        doc_seg_ids[s_idx] = sent_bert_in['seg_ids']
        doc_token_masks[s_idx] = sent_bert_in['token_masks']

    xx = {
        'doc_token_ids': doc_token_ids,
        'doc_seg_ids': doc_seg_ids,
        'doc_token_masks': doc_token_masks,
        'doc_masks': doc_res['doc_mask'],  # max_ns_doc,
    }

    return xx


def build_bert_x_trigger_sl(trigger):
    max_nt = config_model['max_nt_trigger']
    trigger_res = dataset_parser.parse_trigger2words(trigger)
    trigger_bert_in = _build_bert_tokens(words=trigger_res['words'], max_n_tokens=max_nt)

    xx = {
        'trigger_token_ids': trigger_bert_in['token_ids'],
        'trigger_seg_ids': trigger_bert_in['seg_ids'],
        'trigger_token_masks': trigger_bert_in['token_masks'],
        # 'trigger_masks': trigger_res['trigger_mask'],
    }

    return xx


def build_bert_x_sl(trigger, doc_fp):
    xx_doc = build_bert_x_doc_sl(doc_fp)
    xx_trigger = build_bert_x_trigger_sl(trigger)

    return {
        **xx_doc,
        **xx_trigger,
    }
