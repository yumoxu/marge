# -*- coding: utf-8 -*-
import sys
import os
from os.path import join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
parent_sys_path = dirname(sys_path)

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import summ.rank_sent as rank_sent
from utils.config_loader import logger, path_parser
import utils.tools as tools
from data.dataset_parser import dataset_parser
import ir.ir_config as ir_config

import io
from tqdm import tqdm
import numpy as np
import tools.vec_tools as vec_tools
import dill


def load_retrieved_sentences(retrieved_dp, cid):
    """
        For downstream components, e.g., QA model or centrality model.

    :param retrieved_dp:
    :param cid:
    :return:
    """
    if not exists(retrieved_dp):
        raise ValueError('retrieved_dp does not exist: {}'.format(retrieved_dp))

    fp = join(retrieved_dp, cid)
    with io.open(fp, encoding='utf-8') as f:
        content = f.readlines()

    original_sents = [ll.rstrip('\n').split('\t')[-1] for ll in content]
    processed_sents = [dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)
                       for ss in original_sents]

    # processed_sents = []
    # for idx in range(original_sents):
    #     proc_ss = dataset_parser._proc_sent(original_sents[idx], rm_dialog=False, rm_stop=True, stem=True)
    #     if proc_ss:
    #         processed_sents.append(proc_ss)
    #     else:
    #         del original_sents[idx]

    return [original_sents], [processed_sents]  # for compatibility of document organization for similarity calculation


def load_retrieved_passages(cid, get_sents, retrieved_dp=None, passage_ids=None):
    """
        You can retrieve passages from passage_ids
            [1] in retrieved files by setting retrieved_dp OR
            [2] from target passage_ids by setting passage_ids OR
            [3] from all passages by leaving both retrieved_dp and passage_ids to None

    :param cid:
    :param get_sents:
    :param retrieved_dp:
    :param passage_ids:
    :return:
    """
    # if not (retrieved_dp or passage_ids):
    #     raise ValueError('Specify retrieved_dp or passage_ids!')

    passage_ids, passage_fps = tools.get_passage_fps(cid, retrieved_dp=retrieved_dp, passage_ids=passage_ids)

    original_passages, proc_passages = [], []
    for fp in passage_fps:
        with open(fp, 'rb') as f:
            po = dill.load(f)

        if get_sents:
            original_p = po.get_original_sents()
            proc_p = po.get_proc_sents()
        else:
            original_p = po.get_original_passage()
            proc_p = po.get_proc_passage()

        original_passages.append(original_p)
        proc_passages.append(proc_p)

    return original_passages, proc_passages, passage_ids


def load_retrieved_paragraphs(retrieved_dp, cid):
    """
            For downstream components, e.g., QA model or centrality model.
            fixme: this function has been archived.

        :param retrieved_dp:
        :param cid:
        :return:
        """

    if not exists(retrieved_dp):
        raise ValueError('retrieved_dp does not exist: {}'.format(retrieved_dp))

    fp = join(retrieved_dp, cid)
    with io.open(fp, encoding='utf-8') as f:
        content = f.readlines()

    original_paras = [ll.rstrip('\n').split('\t')[-1] for ll in content]

    para_tuples = [dataset_parser._proc_para(pp, rm_dialog=True, rm_stop=True, stem=True, to_str=True)
                       for pp in original_paras]

    original_paras, processed_paras = list(zip(*para_tuples))  # using the new "original" to keep consistency

    return original_paras, processed_paras  # for compatibility of document organization for similarity calculation


def load_retrieved_sentences_with_sid(retrieved_dp, cid):
    """
        For downstream components with sid is needed, e.g., contextual QA model.

    :param retrieved_dp:
    :param cid:
    :return:
    """
    if not exists(retrieved_dp):
        raise ValueError('retrieved_dp does not exist: {}'.format(retrieved_dp))

    fp = join(retrieved_dp, cid)
    with io.open(fp, encoding='utf-8') as f:
        content = f.readlines()

    original_items, processed_items = [], []
    for ll in content:
        items = ll.rstrip('\n').split('\t')
        sid, original_sent = items[0], items[-1]
        processed_sent = dataset_parser._proc_sent(original_sent, rm_dialog=True, rm_stop=True, stem=True)

        original_items.append((sid, original_sent))
        processed_items.append((sid, processed_sent))

    # todo: nest items in again is copied load_retrieved_sentences()
    # todo: it is for compatibility of document organization for similarity calculation
    # todo: double check if it is useful here
    return [original_items], [processed_items]


def load_rank_items(model_name, cid, rank_dp=None):
    if not rank_dp:
        rank_dp = join(path_parser.summary_rank, model_name)
    rank_fp = join(rank_dp, cid)

    with io.open(rank_fp, encoding='utf-8') as f:
        content = f.readlines()

    rank_items = [ll.rstrip('\n').split('\t') for ll in content]
    return rank_items


def _deduplicate(rank_items):
    new_rank_items = []
    sents = []
    for items in rank_items:
        sent = items[-1]
        if sent in sents:
            continue
        sents.append(sent)
        new_rank_items.append(items)

    return new_rank_items


def _norm(rank_items):
    score_list = [float(items[1]) for items in rank_items]

    scores = vec_tools.max_min_scale(scores=np.array(score_list))
    score_list = scores.tolist()
    for i in range(len(rank_items)):
        rank_items[i][1] = str(score_list[i])

    return rank_items, score_list


def _clean(scores):
    """
        No negative scores are allowed in confidence calculation.

        This function simply sets all negative scores to 0.

    """
    return [max(score, 0.0) for score in scores]

def _retrieve_from_rank_items_via_conf(rank_items,
                                       conf_threshold,
                                       deduplicate,
                                       min_ns,
                                       norm=False):
    if deduplicate:
        rank_items = _deduplicate(rank_items)

    conf = 0.0
    if len(rank_items[0]) not in (2, 3):  # 2: w/o sentence; 3: with sentence
        raise ValueError('Corrupted item format: {}'.format(rank_items[0]))

    score_list = [float(items[1]) for items in rank_items]

    if norm:
        rank_items, score_list = _norm(rank_items)

    score_list = _clean(score_list)

    total = sum(score_list)
    retrieved_items = []

    if min_ns:
        n_threshold = min(min_ns, len(rank_items))
    else:
        n_threshold = None

    for items in rank_items:
        retrieved_items.append(items)
        conf += float(items[1]) / total

        if conf >= conf_threshold and (not n_threshold or len(retrieved_items) >= n_threshold):
            break

    return retrieved_items


def _retrieve_from_rank_items_via_comp(rank_items, compression, deduplicate, min_ns):
    if deduplicate:
        rank_items = _deduplicate(rank_items)

    retrieval_size = int(len(rank_items) * compression)

    if min_ns:
        retrieval_size = max(min_ns, retrieval_size)

    return rank_items[:retrieval_size]


def _retrieve_from_rank_items_via_top_k(rank_items, k, deduplicate):
    if deduplicate:
        rank_items = _deduplicate(rank_items)

    return rank_items[:min(len(rank_items), k)]


def _prune_rank_items(rank_items, threshold=1e-10):
    if float(rank_items[-1][1]) > threshold:
        logger.info('[_prune_rank_items] Prune ratio: 0.00')
        return rank_items

    for i in range(len(rank_items)):
        if float(rank_items[i][1]) <= threshold:
            logger.info('[_prune_rank_items] Prune ratio: {0:.2f}'.format(float(i) / len(rank_items)))

            return rank_items[:i]


def retrieve(model_name,
             cid,
             filter_var,
             filter,
             deduplicate,
             min_ns,
             norm=False,
             prune=False,
             rank_dp=None):
    rank_items = load_rank_items(model_name, cid, rank_dp=rank_dp)

    logger.info('cid: {}, #rank_items: {}'.format(cid, len(rank_items)))

    if prune:
        rank_items = _prune_rank_items(rank_items)

    if filter == 'conf':
        retrieved_items = _retrieve_from_rank_items_via_conf(rank_items,
                                                             filter_var,
                                                             deduplicate=deduplicate,
                                                             min_ns=min_ns,
                                                             norm=norm)
    elif filter == 'comp':
        retrieved_items = _retrieve_from_rank_items_via_comp(rank_items,
                                                             filter_var,
                                                             deduplicate=deduplicate,
                                                             min_ns=min_ns)

    elif filter == 'topK':
        retrieved_items = _retrieve_from_rank_items_via_top_k(rank_items,
                                                              filter_var,
                                                              deduplicate=deduplicate)

    else:
        raise ValueError('Invalid FILTER: {}'.format(filter))

    logger.info('retrieved {0}/{1} items for {2}'.format(len(retrieved_items), len(rank_items), cid))

    return retrieved_items


def dump_retrieval(fp, retrieved_items):
    retrieve_records = ['\t'.join(items) for items in retrieved_items]
    n_sents = rank_sent.dump_rank_records(rank_records=retrieve_records, out_fp=fp, with_rank_idx=False)

    logger.info(
        '[dump_retrieve] successfully dumped {0} retrieved items to {1}'.format(n_sents, fp))


def ir_rank2records(model_name, conf_threshold, deduplicate):
    """

    :param model_name:
    :param conf_threshold:
    :param deduplicate:
    :return:
    """
    # todo: add support for compression
    retrieve_dn = ir_config.IR_RECORDS_DIR_NAME_PATTERN.format(model_name, conf_threshold)
    retrieve_dp = join(path_parser.summary_rank, retrieve_dn)

    if exists(retrieve_dp):
        raise ValueError('retrieve_dp exists: {}'.format(retrieve_dp))
    os.mkdir(retrieve_dp)

    cids = tools.get_test_cc_ids()
    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': model_name,
            'cid': cid,
            'filter_var': conf_threshold,
            'filter': 'conf',
            'deduplicate': deduplicate,
        }
        retrieved_items = retrieve(**retrieval_params)
        dump_retrieval(fp=join(retrieve_dp, cid), retrieved_items=retrieved_items)
