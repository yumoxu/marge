from os.path import dirname, abspath
import sys
from utils.config_loader import logger, config_model
from data.clip_and_mask import _clip, _len2mask, mask_para

sys.path.insert(0, dirname(dirname(abspath(__file__))))

"""
    for doc: clip_doc_sents, mask_doc_sents, clip_and_mask_doc_sents
    for trigger (org via sents): clip_trigger_sents, mask_trigger_sents, clip_and_mask_trigger_sents
    for trigger (org via words): clip_trigger, mask_trigger, clip_and_mask_trigger
"""

def clip_doc_sents(sents):
    """
        For QueryNetSL
    :param sents:
    :return:
    """

    words = [ss[:config_model['max_nw_sent']]
             for ss in sents[:config_model['max_ns_doc']]]

    n_words = [len(ss) for ss in words]

    res = {
        'words': words,
        'n_words_by_sents': n_words,
    }

    return res


def mask_doc_sents(n_words):
    """
        mask sentences of a doc with their words.

    :param n_words: an int list of sentence sizes in words.
    """
    mask_shape = (config_model['max_ns_doc'], config_model['max_nw_sent'])
    # logger.info('mask shape: {}'.format(mask_shape))
    return _len2mask(n_words, mask_shape=mask_shape, offset=0)


def clip_and_mask_doc_sents(sents):
    """
        For QueryNetSL.

    :param sents:
    :param offset:
    :return:
    """
    clipped_res = clip_doc_sents(sents)
    # sent_mask = mask_doc_sents(clipped_res['n_words_by_sents'])
    doc_masks = mask_para(len(clipped_res['n_words_by_sents']),
                          max_n_sents=config_model['max_ns_doc'])

    # Warning: keys have been modified as follows, which may cause future errors
    #   'words' => 'sents'
    #   'doc_mask'  => 'doc_masks'
    res = {
        'sents': clipped_res['words'],
        # 'sent_mask': sent_mask,
        'doc_masks': doc_masks,  # max_ns_doc,
    }
    return res


def clip_trigger_sents(sents):
    """
        For QueryNetSL
    :param sents:
    :return:
    """

    sents = [ss[:config_model['max_nw_trigger_sent']] for ss in  sents[:config_model['max_ns_trigger']]]

    n_words = [len(ss) for ss in sents]
    res = {
        'words': sents,
        'n_words_by_sents': n_words,
    }

    return res


def mask_trigger_sents(n_words):
    """
        mask sentences of a doc with their words.

    :param n_words: an int list of sentence sizes in words.
    :param offset: for build mask
    """
    mask_shape = (config_model['max_ns_trigger'], config_model['max_nw_trigger_sent'])
    # logger.info('mask shape: {}'.format(mask_shape))
    return _len2mask(n_words, mask_shape=mask_shape, offset=0)


def clip_and_mask_trigger_sents(sents):
    """
        For QueryNetSL.

    :param sents:
    :return:
    """
    clipped_res = clip_doc_sents(sents)
    # sent_mask = mask_doc_sents(clipped_res['n_words_by_sents'])
    trigger_masks = mask_para(len(clipped_res['n_words_by_sents']), max_n_sents=config_model['max_ns_trigger'])

    res = {
        'sents': clipped_res['sents'],
        'trigger_masks': trigger_masks,  # max_ns_doc,
    }
    return res


def clip_trigger(trigger):
    """
        For QueryNetSL
    :param trigger:
    :return:
    """
    return trigger[:config_model['max_nw_trigger']]


def mask_trigger(nw_trigger):
    mask_shape = [config_model['max_nw_trigger'], ]
    return _len2mask([nw_trigger], mask_shape=mask_shape, offset=0)


def clip_and_mask_trigger(trigger):
    clipped_words = clip_trigger(trigger)
    trigger_mask = mask_trigger(len(clipped_words))

    res = {
        'words': clipped_words,
        'trigger_mask': trigger_mask,
    }

    return res
