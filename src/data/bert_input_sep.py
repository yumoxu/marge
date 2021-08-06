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


def build_bert_x_sep(query, doc_fp):
    # todo: move initial sentence masks here for query and paras
    # build query x
    max_n_query_tokens = config_model['max_n_query_tokens']
    query_res = dataset_parser.parse_query(query)
    query_bert_in = _build_bert_tokens(words=query_res['words'], max_n_tokens=max_n_query_tokens)

    # build para x
    doc_res = dataset_parser.parse_doc(doc_fp, concat_paras=False, offset=1)
    # init paras arrays
    max_n_article_paras = config_model['max_n_article_paras']
    max_n_para_sents = config_model['max_n_para_sents']
    max_n_para_tokens = config_model['max_n_para_tokens']
    basic_para_size = [max_n_article_paras, max_n_para_tokens]

    para_token_ids = np.zeros(basic_para_size, dtype=np.int32)
    para_seg_ids = np.zeros(basic_para_size, dtype=np.int32)
    para_token_masks = np.zeros(basic_para_size)

    # init sentence and para masks
    para_sent_masks = np.zeros([max_n_article_paras, max_n_para_sents, max_n_para_tokens],  dtype=np.float32)
    para_masks = np.zeros([max_n_article_paras, max_n_para_sents], dtype=np.float32)

    # build para
    for para_idx, para_res in enumerate(doc_res['paras']):
        # bert inputs
        para_bert_in = _build_bert_tokens(words=para_res['words'], max_n_tokens=max_n_para_tokens)
        para_token_ids[para_idx] = para_bert_in['token_ids']
        para_seg_ids[para_idx] = para_bert_in['seg_ids']
        para_token_masks[para_idx] = para_bert_in['token_masks']
        # masks
        para_sent_masks[para_idx] = para_res['sent_mask']
        para_masks[para_idx] = para_res['para_mask']

    xx = {
        'query_token_ids': query_bert_in['token_ids'],
        'query_seg_ids': query_bert_in['seg_ids'],
        'query_token_masks': query_bert_in['token_masks'],
        'query_sent_masks': query_res['sent_mask'],
        'query_masks': query_res['para_mask'],

        'para_token_ids': para_token_ids,
        'para_seg_ids': para_seg_ids,
        'para_token_masks': para_token_masks,
        'para_sent_masks': para_sent_masks,
        'para_masks': para_masks,
        'doc_masks': doc_res['doc_masks'],
    }

    return xx
