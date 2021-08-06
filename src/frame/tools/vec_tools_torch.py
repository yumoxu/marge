# -*- coding: utf-8 -*-
import os
from os.path import join, dirname, abspath, exists
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_sim_mat(doc_sent_reps, trigger_sent_reps, score_func):
    """

    :param doc_sent_reps: d_batch * max_ns_doc * d_embed
    :param trigger_sent_reps: d_batch * max_ns_trigger * d_embed
    :param method: cosine, angular, sigmoid
    :return:
         sim_mat: d_batch * max_ns_doc * max_ns_trigger
    """
    if score_func in ('cosine', 'angular'):
        # d_batch = doc_sent_reps.size()[0]
        max_ns_trigger = trigger_sent_reps.size()[1]

        sim_list = []
        # for sample_idx in range(d_batch):
        #     sim = F.cosine_similarity(doc_sent_reps[sample_idx], trigger_sent_reps[sample_idx])
        #     sim_list.append(sim)
        for trigger_s_idx in range(max_ns_trigger):
            trigger_slice = trigger_sent_reps[:, trigger_s_idx:trigger_s_idx+1, :]  # keep dim for broadcast
            sim = F.cosine_similarity(doc_sent_reps, trigger_slice, dim=-1)  # d_batch * max_ns_doc
            sim_list.append(sim)

        sim_mats = torch.stack(sim_list, dim=-1)  # d_batch * max_ns_doc * max_ns_trigger

        if score_func == 'cosine':
            sim_mats = (sim_mats + 1.0) / 2  # in (0, 1)
            return sim_mats

        sim_mats = 1 - torch.acos(sim_mats) / math.pi  # angular
        return sim_mats

    elif score_func in ('tanh', 'sigmoid'):
        # d_embed = doc_sent_reps.size()[-1]
        trigger_sent_reps = trigger_sent_reps.transpose(-1, -2)  # d_batch * d_embed * max_ns_trigger
        score_in = torch.matmul(doc_sent_reps, trigger_sent_reps)  # d_batch * max_ns_doc * max_ns_trigger

        # score_in = score_in / math.sqrt(d_embed)
        if score_func == 'sigmoid':
            sent_scores = torch.sigmoid(score_in)
            return sent_scores

        # else is tanh
        sent_scores = torch.tanh(score_in)
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
    doc_masks = doc_masks.unsqueeze(dim=-1)  # d_batch * max_ns_doc * 1
    trigger_masks = trigger_masks.unsqueeze(dim=1)  # d_batch * 1 * max_ns_trigger

    sim_score_masks = torch.matmul(doc_masks, trigger_masks)  # d_batch * max_ns_doc * max_ns_trigger

    return sim_mat * sim_score_masks


def _compute_relv_scores(sim_mat, pool_func, trigger_masks=None):
    """

    :param sim_mat: d_batch * max_ns_doc * max_ns_trigger
    :param trigger_mask: d_batch * max_ns_trigger
    :return:
        relv_scores: d_batch * max_ns_doc
    """
    if pool_func == 'avg':
        n_query_sents = torch.sum(trigger_masks, dim=-1, keepdim=True)  # d_batch * 1
        nom = torch.sum(sim_mat, dim=-1, keepdim=False)  # d_batch * max_ns_doc

        relv_scores = nom / n_query_sents
        return relv_scores

    elif pool_func == 'max':
        relv_scores = torch.max(sim_mat, dim=-1)[0]
        return relv_scores
    else:
        raise ValueError('Invalid pool_func: {}'.format(pool_func))


def _mask_relv_scores(relv_scores, doc_masks, fill_v=0.0):
    """
        fill padded sentences with score of 0.0 for instance pooling.

    :param relv_scores: d_batch * max_ns_doc
    :param doc_masks: d_batch * max_ns_doc
    :return:
         d_batch * max_ns_doc
    """
    relv_mask = torch.full_like(relv_scores, fill_v)
    relv_scores = torch.where(doc_masks.byte(), relv_scores, relv_mask)  # d_batch * max_ns_doc
    return relv_scores


def get_relv_scores(sent_embeds, trigger_embeds, doc_masks, trigger_masks, score_func, pool_func):
    sim_mat = _compute_sim_mat(sent_embeds, trigger_embeds, score_func=score_func)
    sim_mat = _mask_sim_mat(sim_mat, doc_masks=doc_masks, trigger_masks=trigger_masks)

    # relv_scores: d_batch * max_ns_doc
    relv_scores = _compute_relv_scores(sim_mat, pool_func=pool_func, trigger_masks=trigger_masks)
    relv_scores = _mask_relv_scores(relv_scores, doc_masks=doc_masks)

    return relv_scores
