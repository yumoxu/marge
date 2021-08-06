import utils.config_loader as config
from utils.config_loader import logger, config_model
from data.dataset_parser import dataset_parser
import numpy as np


def _build_bert_tokens_for_para(query_words, para_words):
    token_ids = np.zeros([config_model['max_n_tokens'], ], dtype=np.int32)
    seg_ids = np.zeros([config_model['max_n_tokens'], ], dtype=np.int32)
    token_masks = np.zeros([config_model['max_n_tokens'], ])

    # logger.info('shape of token_ids: {}'.format(token_ids.shape))

    tokens = ['[CLS]'] + query_words + ['[SEP]'] + para_words
    token_id_list = config.bert_tokenizer.convert_tokens_to_ids(tokens)
    n_tokens = len(token_id_list)

    # logger.info('tokens: {}'.format(tokens))
    # logger.info('token_id_list: {}'.format(token_id_list))

    token_ids[:n_tokens] = token_id_list
    seg_ids[len(query_words) + 2:n_tokens] = [1] * len(para_words)
    token_masks[:n_tokens] = [1] * n_tokens

    para_in = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
    }

    return para_in


def build_bert_x(query, doc_fp):
    # prep resources: query and document
    query_res = dataset_parser.parse_query(query)

    para_offset = len(query_res['words']) + 2  # 2 additional tokens for CLS and SEP
    doc_res = dataset_parser.parse_doc(doc_fp, concat_paras=False, offset=para_offset)

    # init arrays
    # token_ids = np.zeros([config_model['max_n_article_paras'], config_model['max_n_tokens']], dtype=np.float32)
    # seg_ids = np.zeros([config_model['max_n_article_paras'], config_model['max_n_tokens']], dtype=np.float32)
    # token_masks = np.zeros([config_model['max_n_article_paras'], config_model['max_n_tokens']], dtype=np.float32)

    token_ids = np.zeros([config_model['max_n_article_paras'], config_model['max_n_tokens']], dtype=np.int32)
    seg_ids = np.zeros([config_model['max_n_article_paras'], config_model['max_n_tokens']], dtype=np.int32)
    token_masks = np.zeros([config_model['max_n_article_paras'], config_model['max_n_tokens']], dtype=np.float32)

    query_sent_masks = np.zeros(
        [config_model['max_n_article_paras'], config_model['max_n_query_sents'], config_model['max_n_tokens']],
        dtype=np.float32)

    para_sent_masks = np.zeros(
        [config_model['max_n_article_paras'], config_model['max_n_para_sents'], config_model['max_n_tokens']],
        dtype=np.float32)

    para_masks = np.zeros([config_model['max_n_article_paras'], config_model['max_n_para_sents']], dtype=np.float32)

    # concat paras with query
    for para_idx, para_res in enumerate(doc_res['paras']):
        # input tokens
        para_in = _build_bert_tokens_for_para(query_words=query_res['words'], para_words=para_res['words'])
        token_ids[para_idx] = para_in['token_ids']
        seg_ids[para_idx] = para_in['seg_ids']
        token_masks[para_idx] = para_in['token_masks']

        # masks
        query_sent_masks[para_idx] = query_res['sent_mask']
        para_sent_masks[para_idx] = para_res['sent_mask']
        para_masks[para_idx] = para_res['para_mask']

    xx = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
        'query_sent_masks': query_sent_masks,
        'query_masks': query_res['para_mask'],
        'para_sent_masks': para_sent_masks,
        'para_masks': para_masks,
        'doc_masks': doc_res['doc_masks'],
    }

    return xx


def build_lexrank_bert_x(sent):
    clipped_words = dataset_parser.parse_lexrank_sent(sent)

    token_ids = np.zeros([config_model['max_n_tokens'], ], dtype=np.int32)
    seg_ids = np.zeros([config_model['max_n_tokens'], ], dtype=np.int32)
    token_masks = np.zeros([config_model['max_n_tokens'], ])

    # logger.info('shape of token_ids: {}'.format(token_ids.shape))

    tokens = ['[CLS]'] + clipped_words
    token_id_list = config.bert_tokenizer.convert_tokens_to_ids(tokens)
    n_tokens = len(token_id_list)

    # logger.info('tokens: {}'.format(tokens))
    # logger.info('token_id_list: {}'.format(token_id_list))

    token_ids[:n_tokens] = token_id_list
    seg_ids[:n_tokens] = [1] * n_tokens
    token_masks[:n_tokens] = [1] * n_tokens

    sent_in = {
        'token_ids': token_ids,
        'seg_ids': seg_ids,
        'token_masks': token_masks,
    }
    return sent_in


def build_paraphrase_bert_x(pos_query, neg_query):
    """

    :param pos_query:
    :param neg_query:
    :return:
        res: a dict containing keys:
            'token_ids',
            'seg_ids'
            'token_masks'
    """
    pos_clipped_words = dataset_parser.parse_paraphrase_sent(pos_query)
    neg_clipped_words = dataset_parser.parse_paraphrase_sent(neg_query)

    res = _build_bert_tokens_for_para(pos_clipped_words, neg_clipped_words)

    paraphrase_in = {
        'input_ids': res['token_ids'],
        'token_type_ids': res['seg_ids'],
        'attention_mask': res['token_masks'],
    }

    return paraphrase_in
