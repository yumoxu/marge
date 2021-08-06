# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

parent_sys_path = dirname(sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

parent_sys_path = dirname(parent_sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import numpy as np
import data.data_pipe_cluster as data_pipe_cluster
from data.dataset_parser import dataset_parser
import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import copy
from torch.autograd import Variable
import utils.tools as tools
import torch
import torch.nn as nn

versions = ['sl', 'alpha']
para_org = True
for vv in versions:
    if config.meta_model_name.endswith(vv):
        para_org = False


def init_model(model, n_iter, restore=None):
    if config_meta['placement'] == 'auto':
        model = nn.DataParallel(model, device_ids=config_meta['device'])

    if config_meta['placement'] in ('auto', 'single'):
        model.cuda()

    checkpoint = join(path_parser.model_save, config.model_name)

    if restore:
        checkpoint = join(checkpoint, 'resume')

    load_checkpoint(checkpoint=checkpoint, model=model, n_iter=n_iter, filter_keys=None)
    model.eval()

    return model


def init_model_and_data(model, n_iter, gen_data_loader, restore=None):
    model = init_model(model, n_iter, restore)

    if gen_data_loader:
        if config_meta['general_specific']:
            data_loader = data_pipe_cluster.GSQSDataLoader()
        else:
            data_loader = data_pipe_cluster.QSDataLoader()
    else:
        data_loader = data_pipe_cluster.QSDataLoaderOneClusterABatch()
    return model, data_loader


def get_sid2score(sent_scores):
    """

    :param sent_scores: d_batch * max_n_paras * max_n_sents
    :return:
    """
    sid2score = dict()
    d_batch, max_n_paras, max_n_sents = sent_scores.shape
    # logger.info('sent_scores.shape: {}'.format(sent_scores.shape))
    for doc_idx in range(d_batch):
        for para_idx in range(max_n_paras):
            for sent_idx in range(max_n_sents):
                value = sent_scores[doc_idx, para_idx, sent_idx]
                # logger.info('value: {}'.format(value))

                if value == -np.inf:
                    break

                key = config.SEP.join((str(doc_idx), str(para_idx), str(sent_idx)))
                sid2score[key] = value
    return sid2score


def get_sid2score_for_one_doc(sent_scores, doc_idx):
    """

    :param sent_scores: d_batch * max_n_paras * max_n_sents
    :return:
    """
    sid2score = dict()
    d_batch, max_n_paras, max_n_sents = sent_scores.shape
    # logger.info('sent_scores.shape: {}'.format(sent_scores.shape))

    if d_batch != 1:
        raise ValueError('Only one doc is allowed in a batch. Invalid d_batch: {}'.format(d_batch))
    sent_scores = sent_scores[0]

    for para_idx in range(max_n_paras):
        for sent_idx in range(max_n_sents):
            value = sent_scores[para_idx, sent_idx]
            # logger.info('value: {}'.format(value))

            if value == -np.inf:
                break

            key = config.SEP.join((str(doc_idx), str(para_idx), str(sent_idx)))
            sid2score[key] = value
    return sid2score


def get_sid2score_for_one_doc_wo_para_org(sent_scores, doc_idx):
    """

    :param sent_scores: d_batch * max_ns_doc
    :return:
    """
    sid2score = dict()
    d_batch, max_ns_doc = sent_scores.shape
    # logger.info('sent_scores.shape: {}'.format(sent_scores.shape))

    if d_batch != 1:
        raise ValueError('Only one doc is allowed in a batch. Invalid d_batch: {}'.format(d_batch))
    sent_scores = sent_scores[0]

    for sent_idx in range(max_ns_doc):
        value = sent_scores[sent_idx]
        # logger.info('value: {}'.format(value))

        if value == -np.inf:
            break

        key = config.SEP.join((str(doc_idx), str(sent_idx)))
        sid2score[key] = value
    return sid2score


def sort_sid2score(sid2score):
    sid_score_list = sorted(sid2score.items(), key=lambda item: item[1], reverse=True)
    return sid_score_list


def archived_get_rank_records(sid_score_list, sents=None):
    """
        Archived. Contains paragraph-organized sid.

        optional: display sentence in record
    :param sid_score_list:
    :param sents:
    :return:
    """
    rank_records = []
    for sid, score in sid_score_list:
        items = [sid, str(score)]
        if sents:
            if type(sents[0][0]) is list:  # organized by paragraphs
                doc_idx, para_idx, sent_idx = tools.get_sent_info(sid)
                sent = sents[doc_idx][para_idx][sent_idx]
            else:
                doc_idx, sent_idx = tools.get_sent_info_sl(sid)
                sent = sents[doc_idx][sent_idx]
            items.append(sent)
        record = '\t'.join(items)
        rank_records.append(record)
    return rank_records


def get_rank_records(sid_score_list, sents=None, flat_sents=False):
    """
        optional: display sentence in record
    :param sid_score_list:
    :param sents:
    :param flat_sents: if True, iterate sent directly; if False, need use sid to get doc_idx and sent_idx.
    :return:
    """
    rank_records = []
    for sid, score in sid_score_list:
        items = [sid, str(score)]
        if sents:
            if flat_sents:
                sent = sents[len(rank_records)]  # the current point
            else:
                logger.info('sid: {}, score: {}'.format(sid, score))
                doc_idx, sent_idx = tools.get_sent_info(sid)
                sent = sents[doc_idx][sent_idx]
                # logger.info('doc_idx.sent_idx: {}.{}, {}'.format(doc_idx, sent_idx, sent))
            items.append(sent)
        record = '\t'.join(items)
        # logger.info('record: {}'.format(record))
        rank_records.append(record)
    return rank_records


def get_rank_records_with_positional_sid(rank_score_list, sents):
    """

    :param rank_score_list:
    :param sents: a list of tuples of (sid, sent)
    :return:
    """
    rank_records = []
    for rank, score in rank_score_list:
        # logger.info('s_rank: {}, score: {}'.format(s_rank, score))
        sid, sent = sents[0][rank]
        items = [sid, str(score), sent]
        record = '\t'.join(items)
        # logger.info('record: {}'.format(record))
        rank_records.append(record)
    return rank_records


def dump_rank_records(rank_records, out_fp, with_rank_idx):
    """
        each line is
            ranking  sid   score

        sid: config.SEP.join((doc_idx, para_idx, sent_idx))
    :param sid_score_list:
    :param out_fp:
    :return:
    """
    lines = rank_records
    if with_rank_idx:
        lines = ['\t'.join((str(rank), record)) for rank, record in enumerate(rank_records)]

    with open(out_fp, mode='a', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return len(lines)


def rank_end2end(model, n_iter, restore=None, with_attn=True):
    model, data_loader = init_model_and_data(model, n_iter, restore)

    rank_dp = join(path_parser.summary_rank, config.model_name)
    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for batch_idx, batch_dict in enumerate(data_loader):
        cid = batch_dict['cid']
        sents = dataset_parser.cid2sents(cid, para_org=True)

        batch = batch_dict['batch']

        feed_dict = copy.deepcopy(batch)

        for (k, v) in feed_dict.items():
            with torch.no_grad():
                feed_dict[k] = Variable(v, requires_grad=False)

        model_res = model(**feed_dict)

        # turn vars to numpy arrays

        # doc_scores = model_res['doc_scores'].data.cpu().numpy()
        # y_preds = metrics.get_y_pred_2d(doc_scores, max_alter=True)  # todo: fix this if use
        y_true = batch['yy'].cpu().numpy()
        sent_scores = model_res['sent_scores'].data.cpu().numpy()  # d_batch * max_n_paras * max_n_sents

        if with_attn:
            sent_attn = model_res['sent_attn'].data.cpu().numpy()  # d_batch * max_n_paras * max_n_sents

            # weight sent scores with their attn, and mask
            sent_scores = np.multiply(sent_scores, sent_attn)  # d_batch * max_n_paras * max_n_sents
            para_masks = batch['para_masks'].cpu().numpy()  # d_batch * max_n_paras * max_n_sents
            mask_v = np.full_like(sent_scores, -np.inf)
            sent_scores = np.where(para_masks, sent_scores, mask_v)  # d_batch * max_n_paras * max_n_sents

        logger.info('[RANKING] creating sid2score')
        sid2score = get_sid2score(sent_scores)

        logger.info('[RANKING] ranking sentence scores')
        sid_score_list = sort_sid2score(sid2score)
        rank_records = get_rank_records(sid_score_list, sents=sents)

        out_fp = join(rank_dp, cid)
        logger.info('[RANKING] dumping ranking file to: {0}'.format(out_fp))
        n_sents = dump_rank_records(rank_records, out_fp=out_fp)
        logger.info('[RANKING] successfully dumped ranking of {0} sentences from {1} documents'.format(n_sents,
                                                                                                       y_true.shape[0]))


def rank_end2end_one_doc_a_batch(model, n_iter, restore=None):
    model, data_loader_generator = init_model_and_data(model, n_iter, gen_data_loader=True, restore=restore)

    rank_dp = tools.get_rank_dp(config.model_name, n_iter=n_iter)

    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for data_loader in data_loader_generator:
        sid2score = dict()
        logger.info('[RANKING] creating sid2score')
        for doc_idx, batch in enumerate(data_loader):
            feed_dict = copy.deepcopy(batch)

            for (k, v) in feed_dict.items():
                with torch.no_grad():
                    feed_dict[k] = Variable(v, requires_grad=False)

            model_res = model(**feed_dict)

            # turn vars to numpy arrays
            # doc_scores = model_res['doc_scores'].data.cpu().numpy()
            # y_preds = metrics.get_y_pred_2d(doc_scores, max_alter=True)  # todo: fix this if use
            # y_true = batch['yy'].cpu().numpy()
            sent_scores = model_res['sent_scores'].data.cpu().numpy()  # d_batch * max_n_paras * max_n_sents

            if not para_org:
                sent_scores = 1 - sent_scores

            # mask
            if not para_org:
                score_masks = batch['doc_masks'].cpu().numpy()  # d_batch * max_ns_doc
            else:
                score_masks = batch['para_masks'].cpu().numpy()  # d_batch * max_n_paras * max_n_sents

            mask_v = np.full_like(sent_scores, -np.inf)
            sent_scores = np.where(score_masks, sent_scores, mask_v)  # d_batch * max_n_paras * max_n_sents

            if not para_org:
                new_sid2score = get_sid2score_for_one_doc_wo_para_org(sent_scores, doc_idx)
            else:
                new_sid2score = get_sid2score_for_one_doc(sent_scores, doc_idx)

            sid2score = {
                **new_sid2score,
                **sid2score,
            }

        logger.info('[RANKING] ranking sentence scores')
        sid_score_list = sort_sid2score(sid2score)

        sents = dataset_parser.cid2sents(cid=data_loader.cid, para_org=para_org)
        rank_records = get_rank_records(sid_score_list, sents=sents)

        out_fp = join(rank_dp, data_loader.cid)
        logger.info('[RANKING] dumping ranking file to: {0}'.format(out_fp))
        n_sents = dump_rank_records(rank_records, out_fp=out_fp)
        logger.info('[RANKING] successfully dumped ranking of {0} sentences for {1}'.format(n_sents, data_loader.cid))
