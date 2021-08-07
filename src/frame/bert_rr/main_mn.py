# -*- coding: utf-8 -*-
import sys
from os.path import isfile, isdir, join, dirname, abspath, exists
sys_path = dirname(dirname(abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

parent_sys_path = dirname(sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

parent_sys_path = dirname(parent_sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import utils.config_loader as config

from utils.config_loader import logger, path_parser, config_meta, meta_model_name
from pathlib import Path

import copy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.tools as tools

from argparse import ArgumentParser
import os
import io
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import shutil

import summ.rank_sent as rank_sent
import summ.select_sent as select_sent
import summ.compute_rouge as rouge
import ir.ir_tools as ir_tools
from bert_rr.data_pipe_mn import (QSDataLoader, get_test_cc_ids, load_retrieved_sentences)
import bert_rr.rr_config as rr_config

from unilm_utils.unilm_eval import UniLMEval
from unilm_utils.unilm_input import UniLMInput


"""
    This module builds the following pipeline:

    rel_scores [compute, dump] =>
    rr_rank [load rel_scores, compute, dump]=>
    rr_records [load rr_rank, compute, dump]

    For the same IR records and query type, rel_scores and rr_rank only need to be processed once.

    rr_records can be generated with different threhold, e.g., confidence-based (CONF_THRESHOLD_RR) or TopK-based (TOP_NUM_RR).

    tune() can tune threholds. Also, you can use it to produce TopK Recall Curve, which can be used to evaluate Semantic Matching Model.

"""

N_SUMMARY_SENTS = 10 
rel_scores_dp = path_parser.graph_rel_scores / rr_config.RELEVANCE_SCORE_DIR_NAME_MN
rank_dp = path_parser.summary_rank / rr_config.RR_MODEL_NAME_BERT_MN
ir_rec_dp = path_parser.data / 'test_mn_cluster'  # before: test_mn


def init():
    # parse args
    parser = ArgumentParser()
    parser.add_argument('n_devices',
                        nargs='?',
                        default=4,
                        help='num of devices on which model will be running on')

    args = parser.parse_args()
    all_device_ids = [0, 1, 2, 3]
    device = all_device_ids[:int(args.n_devices)]
    # device = [int(d) for d in args.n_devices]
    config_meta['device'] = device

    if not torch.cuda.is_available():
        placement = 'cpu'
        logger.info('[MAIN INIT] path mode: {0}, placement: {1}'.format(config.path_type, placement))
    else:
        if len(device) == 1:
            placement = 'single'
            torch.cuda.set_device(device[0])
        elif config_meta['auto_parallel']:
            placement = 'auto'
        else:
            placement = 'manual'

        logger.info(
            '[MAIN INIT] path mode: {0}, placement: {1}, n_devices: {2}'.format(config.path_type, placement,
                                                                                args.n_devices))
    config_meta['placement'] = placement


def _place_model(model):
    # epoch, model, tokenizer, scores = load_checkpoint()
    if config_meta['placement'] == 'auto':
        model = nn.DataParallel(model, device_ids=config_meta['device'])
        logger.info('[place_model] Parallel Data to devices: {}'.format(config_meta['device']))

    if config_meta['placement'] in ('auto', 'single'):
        model.cuda()

    model.eval()
    return model


def _dump(model, cluster_loader, dump_dp):
    doc_rel_scores = []
    for _, batch in enumerate(cluster_loader):
        feed_dict = copy.deepcopy(batch)

        for (k, v) in feed_dict.items():
            with torch.no_grad():
                feed_dict[k] = Variable(v, requires_grad=False)

        n_sents, max_nt = feed_dict['token_ids'].size()
        # pred: (batch * max_ns_doc) * 2
        pred = model(feed_dict['token_ids'],
                     feed_dict['seg_ids'],
                     feed_dict['token_masks'])

        if type(pred) is tuple:  # BertForSequenceClassification returns tuple
            pred = pred[0]

        n_cls = pred.size()[-1]
        if n_cls == 2:
            pred = F.softmax(pred, dim=-1)[:, 1]
        elif n_cls == 1:
            pred = pred.squeeze(-1)
        else:
            raise ValueError('Invalid n_cls: {}'.format(n_cls))

        rel_scores = pred.cpu().detach().numpy()  # d_batch,
        logger.info('[_dump] rel_scores: {}'.format(rel_scores.shape))

        doc_rel_scores.append(rel_scores[:n_sents])

    rel_scores = np.concatenate(doc_rel_scores)

    dump_fp = join(dump_dp, cluster_loader.cid)

    tools.save_obj(obj=rel_scores, fp=dump_fp)
    logger.info('[_dump] dumping ranking file to: {0}'.format(dump_fp))


def dump_rel_scores():
    if exists(rel_scores_dp):
        raise ValueError('rel_scores_dp exists: {}'.format(rel_scores_dp))
    os.mkdir(rel_scores_dp)

    model = _place_model(model=config.bert_model)
    data_gen = QSDataLoader(retrieve_dp=ir_rec_dp, 
        n_summary_sents=N_SUMMARY_SENTS,
        no_query=rr_config.NO_QUERY)

    for cluster_loader in data_gen:
        if config.meta_model_name in ('bert_rr'):
            _dump(model, cluster_loader=cluster_loader, dump_dp=rel_scores_dp)
        else:
            raise ValueError('Invalid meta_model_name: {}'.format(config.meta_model_name))


def load_rel_scores(cid, rel_scores_dp):
    rel_scores_fp = join(rel_scores_dp, cid)
    return tools.load_obj(rel_scores_fp)


def rel_scores2rank():
    cids = get_test_cc_ids(rel_scores_dp)
    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    for cid in tqdm(cids):
        rel_scores = load_rel_scores(cid=cid, rel_scores_dp=rel_scores_dp)
        sent_ids = np.argsort(rel_scores)[::-1].tolist()

        sid_score_list = []
        for sid in sent_ids:
            sid_score = ('0_{}'.format(sid), rel_scores[sid])
            sid_score_list.append(sid_score)

        original_sents, _ = load_retrieved_sentences(retrieved_dp=ir_rec_dp, cid=cid)
        # sentences = [[sent.replace('NEWLINE_CHAR', '').strip() for sent in original_sents[0]]]
        rank_records = rank_sent.get_rank_records(sid_score_list, sents=original_sents)

        n_sents = rank_sent.dump_rank_records(rank_records=rank_records, out_fp=join(rank_dp, cid), with_rank_idx=False)
        logger.info('Dump {} ranking records'.format(n_sents))


def tune():
    """
        Tune RR confidence / compression rate / topK
        based on Recall Rouge 2.
    :return:
    """
    if rr_config.FILTER in ('conf', 'comp'):
        tune_range = np.arange(0.05, 1.05, 0.05)
    else:  # topK
        interval = 10
        if rr_config.ir_config.FILTER == 'topK':
            end = rr_config.ir_config.FILTER_VAR + interval
        else:
            end = 200 + interval
        tune_range = range(interval, end, interval)

    rr_tune_dp = join(path_parser.summary_rank, rr_config.RR_TUNE_DIR_NAME_BERT_MN)
    rr_tune_result_fp = join(path_parser.tune, rr_config.RR_TUNE_DIR_NAME_BERT_MN)
    with open(rr_tune_result_fp, mode='a', encoding='utf-8') as out_f:
        headline = 'Filter\tRecall\tF1\n'
        out_f.write(headline)

    cids = tools.get_test_cc_ids()
    for filter_var in tune_range:
        if exists(rr_tune_dp):  # remove previous output
            shutil.rmtree(rr_tune_dp)
        os.mkdir(rr_tune_dp)

        for cid in tqdm(cids):
            retrieval_params = {
                'model_name': rr_config.RR_MODEL_NAME_BERT_MN,
                'cid': cid,
                'filter_var': filter_var,
                'filter': rr_config.FILTER,
                'deduplicate': None,
                'min_ns': rr_config.RR_MIN_NS
            }

            if meta_model_name.startswith('bert_squad') and config.squad_var.startswith('bert_shared'):
                retrieval_params['norm'] = True

            retrieved_items = ir_tools.retrieve(**retrieval_params)
            summary = '\n'.join([item[-1] for item in retrieved_items])
            # print(summary)
            with open(join(rr_tune_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(summary)

        performance = rouge.compute_rouge_for_dev(rr_tune_dp, tune_centrality=False)
        with open(rr_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if rr_config.FILTER in ('conf', 'comp'):
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = '{0}\t{1}\n'.format(filter_var, performance)

            out_f.write(rec)

    if exists(rr_tune_dp):  # remove previous output
        shutil.rmtree(rr_tune_dp)


def finer_tune():
    """
        Tune RR confidence / compression rate / topK
        based on Recall Rouge 2.
    :return:
    """
    if rr_config.FILTER in ('conf', 'comp'):
        tune_range = np.arange(0.05, 1.05, 0.05)
    else:  # topK
        tune_range = range(1, 31)

    rr_tune_dp = path_parser.summary_rank / rr_config.RR_FINER_TUNE_DIR_NAME_BERT_MN
    rr_tune_result_fp = path_parser.tune / f'{rr_config.RR_FINER_TUNE_DIR_NAME_BERT_MN}.txt'
    with open(rr_tune_result_fp, mode='a', encoding='utf-8') as out_f:
        headline = 'Filter\tRecall\tF1\n'
        out_f.write(headline)

    cids = tools.get_test_cc_ids()
    for filter_var in tune_range:
        if exists(rr_tune_dp):  # remove previous output
            shutil.rmtree(rr_tune_dp)
        os.mkdir(rr_tune_dp)

        for cid in tqdm(cids):
            retrieval_params = {
                'model_name': rr_config.RR_MODEL_NAME_BERT_MN,
                'cid': cid,
                'filter_var': filter_var,
                'filter': rr_config.FILTER,
                'deduplicate': None,
                'min_ns': rr_config.RR_MIN_NS
            }

            if meta_model_name.startswith('bert_squad') and config.squad_var.startswith('bert_shared'):
                retrieval_params['norm'] = True

            retrieved_items = ir_tools.retrieve(**retrieval_params)
            summary = '\n'.join([item[-1] for item in retrieved_items])
            # print(summary)
            with open(join(rr_tune_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(summary)

        performance = rouge.compute_rouge_for_dev(rr_tune_dp, tune_centrality=False)
        with open(rr_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if rr_config.FILTER in ('conf', 'comp'):
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = '{0}\t{1}\n'.format(filter_var, performance)

            out_f.write(rec)

    if exists(rr_tune_dp):  # remove previous output
        shutil.rmtree(rr_tune_dp)


def rr_rank2records():
    rr_rec_dp = join(path_parser.summary_rank, rr_config.RR_RECORD_DIR_NAME_BERT_MN)

    if exists(rr_rec_dp):
        raise ValueError('rr_rec_dp exists: {}'.format(rr_rec_dp))
    os.mkdir(rr_rec_dp)

    cids = get_test_cc_ids(rel_scores_dp)
    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': rr_config.RR_MODEL_NAME_BERT_MN,
            'cid': cid,
            'filter_var': rr_config.FILTER_VAR,
            'filter': rr_config.FILTER,
            'deduplicate': None,
            'min_ns': rr_config.RR_MIN_NS
        }
        
        # todo: examine if we want to normalize scores for bert_rr
        if meta_model_name.startswith('bert_squad') and config.squad_var.startswith('bert_shared'):
            retrieval_params['norm'] = True

        retrieved_items = ir_tools.retrieve(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(rr_rec_dp, cid), retrieved_items=retrieved_items)


def select_e2e():
    """
        This function is for ablation study (w/o Centrality).

    """
    cc_ids = get_test_cc_ids(rel_scores_dp)
    params = {
        'model_name': rr_config.RR_MODEL_NAME_BERT_MN,
        'cos_threshold': 0.6,
        'retrieved_dp': ir_rec_dp,
        'cc_ids': cc_ids,
    }
    select_sent.select_end2end_mn(**params)


def compute_rouge():
    text_params = {
        'model_name': rr_config.RR_MODEL_NAME_BERT_MN,
        'cos_threshold': 0.6,
        'n_iter': None,
        'diversity_param_tuple': None,
        'extra': None,
    }

    text_dp = tools.get_text_dp(**text_params)
    output = rouge.compute_rouge_for_mn(text_dp)
    return output


def get_rank_dp(positional):
    """
        Copied from bert_marge/main.py.

    """
    dn = rr_config.RR_MODEL_NAME_BERT
    if positional:
        dn += '-w_pos'
    
    rank_dp = path_parser.summary_rank / dn
    return rank_dp


def get_text_dp():
    """
        Copied from bert_marge/main.py.
        
    """

    assert rr_config.USE_CENTRALITY
    dn = rr_config.CENTRALITY_DIR_NAME
    text_dp = path_parser.summary_text / dn
    return text_dp


def build_unilm_input(src):
    """
        Copied from bert_marge/main.py.
        
    """
    rank_dp = None
    text_dp = None

    if src == 'rank':
        rank_dp = get_rank_dp(positional=True)
    elif src == 'text':
        text_dp = get_text_dp()
    
    cids = get_test_cc_ids(rel_scores_dp)
    unilm_input = UniLMInput(marge_config=rr_config,
        rank_dp=rank_dp,
        text_dp=text_dp, 
        fix_input=True, 
        cluster_ids=cids)

    if src == 'rank':
        unilm_input.build_from_rank()
    elif src == 'text':
        unilm_input.build_from_text()


def eval_unilm_out():
    """
        Copied from bert_marge/main.py.
        
    """
    cids = get_test_cc_ids(rel_scores_dp)
    unilm_eval = UniLMEval(marge_config=rr_config, 
        pre_tokenize_sent=False, 
        max_eval_len=300, 
        cluster_ids=cids,
        eval_mn=True)
    # unilm_eval.build_and_eval_unilm_out()
    unilm_eval.eval_unilm_out()


if __name__ == '__main__':
    init()
    dump_rel_scores()
    rel_scores2rank()

    # tune()  # select div hyper-parameter
    # finer_tune()  # visualize n_sents

    rr_rank2records()  # with selected hyper-parameter
    select_e2e()
    compute_rouge()

    # build_unilm_input(src='text')
    # eval_unilm_out()
