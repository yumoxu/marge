# -*- coding: utf-8 -*-
import os
from os.path import join, dirname, abspath, exists
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
import sklearn
import numpy as np
from scipy.stats import logistic
import math


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def _compute_sim_mat(doc_sent_reps, trigger_sent_reps, score_func):
    """

    :param doc_sent_reps: d_batch * max_ns_doc * d_embed
    :param trigger_sent_reps: d_batch * max_ns_trigger * d_embed
    :param method: cosine, angular, sigmoid
    :return:
         sim_mat: d_batch * max_ns_doc * max_ns_trigger
    """
    if score_func in ('cosine', 'angular'):
        d_batch = doc_sent_reps.shape[0]

        sim_mats = [sklearn.metrics.pairwise.cosine_similarity(doc_sent_reps[sample_idx], trigger_sent_reps[sample_idx])
                    for sample_idx in range(d_batch)]

        sim_mats = np.stack(sim_mats, axis=0)
        if score_func == 'cosine':
            return sim_mats

        sim_mats = 1 - np.arccos(sim_mats) / math.pi  # angular
        return sim_mats

    elif score_func == 'tanh':
        d_embed = doc_sent_reps.shape[-1]
        query_sent_reps = np.transpose(trigger_sent_reps, [0, 2, 1])  # d_batch * d_embed * max_ns_trigger
        score_in = np.matmul(doc_sent_reps, query_sent_reps)  # d_batch * max_ns_doc * max_ns_trigger

        score_in /= math.sqrt(d_embed)
        sent_scores = np.tanh(score_in)

        return sent_scores

    elif score_func == 'sigmoid':
        d_embed = doc_sent_reps.shape[-1]
        query_sent_reps = np.transpose(trigger_sent_reps, [0, 2, 1])  # d_batch * d_embed * max_ns_trigger
        score_in = np.matmul(doc_sent_reps, query_sent_reps)  # d_batch * max_ns_doc * max_ns_trigger

        score_in /= math.sqrt(d_embed)
        sent_scores = logistic.cdf(score_in)
        return sent_scores

    else:
        raise ValueError('Invalid method: {}'.format(score_func))


def _mask_sim_mat(sim_mat, doc_masks, trigger_masks):
    """

    :param sim_mat: d_batch * max_ns_doc * max_ns_trigger
    :param doc_masks: d_batch * max_ns_doc
    :param trigger_masks: d_batch * max_ns_trigger
    :return:
    """
    doc_masks = np.expand_dims(doc_masks, axis=-1)  # d_batch * max_ns_doc * 1
    trigger_masks = np.expand_dims(trigger_masks, axis=1)  # d_batch * 1 * max_ns_trigger

    sim_score_masks = np.matmul(doc_masks, trigger_masks)  # d_batch * max_ns_doc * max_ns_trigger

    return sim_mat * sim_score_masks


def _compute_relv_scores(sim_mat, pool_func, trigger_masks=None):
    """

    :param sim_mat: d_batch * max_ns_doc * max_ns_trigger
    :param trigger_mask: d_batch * max_ns_trigger
    :return:
        relv_scores: d_batch * max_ns_doc
    """
    if pool_func == 'avg':
        n_query_sents = np.sum(trigger_masks, axis=-1, keepdims=True)  # d_batch * 1
        nom = np.sum(sim_mat, axis=-1)  # d_batch * max_ns_doc

        relv_scores = nom / n_query_sents
        return relv_scores

    elif pool_func == 'max':
        relv_scores = np.max(sim_mat, axis=-1)
        return relv_scores
    else:
        raise ValueError('Invalid pool_func: {}'.format(pool_func))


def _mask_relv_scores(relv_scores, doc_masks):
    """
        fill padded sentences with score of -np.inf for ranking purpose.

    :param relv_scores: d_batch * max_ns_doc
    :param doc_masks: d_batch * max_ns_doc
    :return:
         d_batch * max_ns_doc
    """

    relv_mask = np.full_like(relv_scores, -np.inf)
    relv_scores = np.where(doc_masks, relv_scores, relv_mask)  # d_batch * max_ns_doc
    return relv_scores


def get_relv_scores(sent_embeds, trigger_embeds, doc_masks, trigger_masks, score_func, pool_func):
    sim_mat = _compute_sim_mat(sent_embeds, trigger_embeds, score_func=score_func)
    sim_mat = _mask_sim_mat(sim_mat, doc_masks=doc_masks, trigger_masks=trigger_masks)

    # relv_scores: d_batch * max_ns_doc
    relv_scores = _compute_relv_scores(sim_mat, pool_func=pool_func, trigger_masks=trigger_masks)
    relv_scores = _mask_relv_scores(relv_scores, doc_masks=doc_masks)

    return relv_scores


def max_min_scale(scores):
    """

    :param scores: could be a vector or a matrix.
    :return:
    """
    min_v = np.min(scores)
    max_v = np.max(scores)
    denom = max_v - min_v
    scores = (scores - min_v) / denom
    return scores


def _clean(rel_scores):
    """
        No negative scores are allowed in confidence calculation.

        This function simply sets all negative scores to 0.

    """
    return rel_scores.clip(min=0)


def norm_rel_scores(rel_scores, max_min_scale, passage_proc=False):
    """
        The transition matrix requires all elements to be in [0, 1] and all rows sum to 1.

        We use max-min scale + l1 norm for both sim_mat and rel_vec, to achieve this.

        Note:
        max-min scale was not adopted by Wan's paper on query based text summarization; it is not necessary since all TFIDF scores are positive.


    :param rel_scores: in [0, 1]
    :param passage_proc: for bert_passage only
    :return:
    """
    if max_min_scale:
        rel_scores = max_min_scale(rel_scores)

    if passage_proc:
        rel_scores = np.sqrt(rel_scores)
        rel_scores = np.tanh(rel_scores)

    rel_scores = _clean(rel_scores)  # set negative scores to 0
    
    rel_scores = rel_scores / np.sum(rel_scores)  # l1 norm to make a distribution
    return rel_scores


def norm_sim_mat(sim_mat, max_min_scale):
    """
        The transition matrix requires all elements to be in [0, 1] and all rows sum to 1.

        We use max-min scale + l1 norm for both sim_mat and rel_vec, to achieve this.

        Note:
        (1) max-min scale was not adopted by Wan's paper on query based text summarization; we believe it is a flaw.
        (2) l1 norm for sim_mat is implemented in lexrank package; here we just max-min scale it.


    :param sim_mat:
    :return:
    """
    np.fill_diagonal(sim_mat, 0.0)  # avoid self-transition
    # deal with row_sum == 0.0. Not needed as per Wan's paper.
    # set a uniform number
    # row_sum = sim_mat.sum(axis=1, keepdims=True)
    # cond = row_sum == 0.0
    # zeros = np.zeros_like(row_sum)
    # fill_number = 1.0/(len(sim_mat)-1)  # make non-self elements add to 1.0
    # fill = np.full_like(row_sum, fill_number)
    # extra = np.where(cond, fill, zeros)
    # sim_mat += extra
    # np.fill_diagonal(sim_mat, 0.0)  # remove self
    if max_min_scale:  # todo: check if scale should come first than fill_diagonal
        sim_mat = max_min_scale(sim_mat)

    return sim_mat
