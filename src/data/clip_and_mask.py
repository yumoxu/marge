from os.path import dirname, abspath
import sys
import utils.tools as tools
from utils.config_loader import logger, config_model
import numpy as np

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def _get_max_n_tokens(src, join_query_para):
    if join_query_para or src == 'whole':
        return config_model['max_n_tokens']

    if src == 'q':
        return config_model['max_n_query_tokens']
    elif src == 'p':
        return config_model['max_n_para_tokens']
    else:
        raise ValueError('Invalid src: {}'.format(src))


def _clip(sents, max_n_sents, max_n_words, flat=True):
    """

    :param parsed_para: a list of sentences. each sentence is a list of words.
    :return:
    """
    sents = sents[:max_n_sents]

    n_words = [len(ss) for ss in sents]

    ns = len(sents)
    origin_nw = sum(n_words)

    if origin_nw <= max_n_words:
        if flat:
            sents = tools.flatten(sents)
        res = {
            'words': sents,
            'n_words_by_sents': n_words,
        }
        return res

    clip_idx, spare = None, None

    for idx in range(1, ns+1):
        clip_idx = -idx
        spare = max_n_words - sum(n_words[:clip_idx])
        if spare >= 0:
            # logger.info('spare: {}'.format(spare))
            break

    clipped = sents[:clip_idx]
    spare_items = sents[clip_idx][:spare]
    if spare_items:
        clipped.append(spare_items)

    n_words = [len(ss) for ss in clipped]

    if sum(n_words) != max_n_words:
        logger.error('original: {}'.format(sents))
        logger.error('clipped: {}'.format(clipped))
        raise ValueError(
            'sum(n_words): {0} != max_n_words :{1}. len before append last: {2}'.format(sum(n_words), max_n_words,
                                                                                        sum(n_words[:-1])))

    if flat:
        clipped = tools.flatten(clipped)

    res = {
        'words': clipped,
        'n_words_by_sents': n_words,
    }

    return res


def clip_query_sents(sents):
    return _clip(sents, config_model['max_n_query_sents'], config_model['max_n_query_words'])


def clip_para_sents(sents):
    return _clip(sents, config_model['max_n_para_sents'], config_model['max_n_para_words'])


def clip_article_paras(paras):
    return paras[:config_model['max_n_article_paras']]


def clip_lexrank_sent_words(words):
    return words[:config_model['max_n_tokens']-1]  # 1 is for CLS token


def clip_paraphrase_sent_words(words):
    max_n_words = int((config_model['max_n_tokens'] - 2) / 2)  # 2 is for CLS and SEP
    return words[:max_n_words]


def _len2mask(lens, mask_shape, offset):
    """
        could be applied to:
            [1] paragraph masks: pooling paragraph instance scores to document bag score
            [2] sentence masks: pooling word representations to sentence representation
    :param lens: n_sents or n_paras
    :param mask_shape: [max_n_sents, max_words] or [max_n_docs, max_n_paras]
    :return:
    """
    mask = np.zeros(mask_shape, dtype=np.float32)
    # logger.info('mask shape: {}'.format(mask.shape))
    if type(lens) != list:
        raise ValueError('Invalid lens type: {}'.format(type(lens)))

    if len(mask_shape) == 1:  # mask a document with its paras
        mask[offset:offset + lens[0]] = [1] * lens[0]
        return mask

    elif len(mask_shape) == 2:  # mask sentences of a para/query with their words
        for idx, ll in enumerate(lens):
            end = offset + ll
            # logger.info('offset: {0}, end: {1}'.format(offset, end))
            # logger.info('shape: {0}, #input: {1}'.format(mask[idx, start:end].shape, ll))
            mask[idx, offset:end] = [1] * ll
            offset = end
        return mask

    else:
        raise ValueError('Invalid mask dim: {}'.format(len(mask_shape)))


def mask_query_sents(n_words, offset, join_query_para):
    """
        mask sentences of a query with their words.

    :param n_words: an int list of sentence sizes in words.
    :return:
    """
    max_n_tokens = _get_max_n_tokens(src='q', join_query_para=join_query_para)
    mask_shape = (config_model['max_n_query_sents'], max_n_tokens)
    return _len2mask(n_words, mask_shape=mask_shape, offset=offset)  # the first token is [CLS]


def mask_para_sents(n_words, offset, join_query_para):
    """
        mask sentences of a para with their words.

    :param n_words: an int list of sentence sizes in words.
    :param offset: for build mask
    """
    max_n_tokens = _get_max_n_tokens(src='p', join_query_para=join_query_para)
    mask_shape = (config_model['max_n_para_sents'], max_n_tokens)
    # logger.info('mask shape: {}'.format(mask_shape))
    return _len2mask(n_words, mask_shape=mask_shape, offset=offset)


def mask_para(n_sents, max_n_sents):
    """

    :param n_sents: an int.
    :param max_n_sents:
    :return:
    """
    mask_shape = [max_n_sents, ]
    return _len2mask([n_sents], mask_shape=mask_shape, offset=0)


def mask_doc(n_paras):
    """
        mask an article with its paras.

    :param n_paras: an int of article size in paras.
    :return:
    """
    mask_shape = [config_model['max_n_article_paras'], ]
    return _len2mask([n_paras], mask_shape=mask_shape, offset=0)


def mask_lexrank_sent(n_words):
    """
        mask a sentence with its words.
    :param n_words: an int of sentence size in words.
    :return:
    """
    mask_shape = [config_model['max_n_tokens'], ]
    return _len2mask([n_words], mask_shape=mask_shape, offset=0)


def clip_and_mask_query_sents(sents, offset=1, join_query_para=True):
    clipped_res = clip_query_sents(sents)
    # logger.info('para clipped len: {}'.format(sum(clipped_res['n_words_by_sents'])))
    sent_mask = mask_query_sents(clipped_res['n_words_by_sents'], offset=offset, join_query_para=join_query_para)
    para_mask = mask_para(len(clipped_res['n_words_by_sents']), max_n_sents=config_model['max_n_query_sents'])

    res = {
        'words': clipped_res['words'],
        'sent_mask': sent_mask,
        'para_mask': para_mask,
    }
    return res


def clip_and_mask_para_sents(sents, offset, join_query_para=True):
    clipped_res = clip_para_sents(sents)
    sent_mask = mask_para_sents(clipped_res['n_words_by_sents'], offset=offset, join_query_para=join_query_para)
    para_mask = mask_para(len(clipped_res['n_words_by_sents']), max_n_sents=config_model['max_n_para_sents'])

    res = {
        'words': clipped_res['words'],
        'sent_mask': sent_mask,
        'para_mask': para_mask,
    }
    return res


def clip_and_mask_a_doc(paras):
    clipped_paras = clip_article_paras(paras)
    doc_masks = mask_doc(len(clipped_paras))

    res = {
        'paras': clipped_paras,
        'doc_masks': doc_masks,
    }

    return res


def clip_and_mask_a_lexrank_sent(words):
    clipped_words = clip_lexrank_sent_words(words)
    sent_masks = mask_lexrank_sent(len(clipped_words))

    res = {
        'words': clipped_words,
        'sent_masks': sent_masks,
    }

    return res
