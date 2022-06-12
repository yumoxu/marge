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

import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_meta, meta_model_name
from os import listdir
from os.path import isfile, isdir, join, dirname, abspath, exists


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
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent
import ir.ir_tools as ir_tools
# from ir.ir_tools import load_retrieved_sentences
from bert_rr.data_pipe_cluster import QSDataLoader, load_retrieved_sentences
import bert_rr.rr_config as rr_config
from sklearn.metrics.pairwise import cosine_similarity
import shutil
import summ.compute_rouge as rouge

from frame.unilm_utils.unilm_eval import UniLMEval
from frame.unilm_utils.unilm_input import UniLMInput

import tools.general_tools as general_tools

"""
    This module builds the following pipeline:

    rel_scores [compute, dump] =>
    rr_rank [load rel_scores, compute, dump]=>
    rr_records [load rr_rank, compute, dump]

    For the same IR records and query type, rel_scores and rr_rank only need to be processed once.

    rr_records can be generated with different threhold, e.g., confidence-based (CONF_THRESHOLD_RR) or TopK-based (TOP_NUM_RR).

    tune() can tune threholds. Also, you can use it to produce TopK Recall Curve, which can be used to evaluate Semantic Matching Model.

"""

rel_scores_dp = join(path_parser.graph_rel_scores, rr_config.RELEVANCE_SCORE_DIR_NAME)
cids = tools.get_test_cc_ids()

rank_dp = join(path_parser.summary_rank, rr_config.RR_RANK_DIR_NAME_BERT)
ir_rec_dp = join(path_parser.summary_rank, rr_config.IR_RECORDS_DIR_NAME)


def init():
    # parse args
    parser = ArgumentParser()
    parser.add_argument('n_devices',
                        nargs='?',
                        default=4,
                        help='num of devices on which model will be running on')

    args = parser.parse_args()
    all_device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    device = all_device_ids[:int(args.n_devices)]
    # device = [int(d) for d in args.n_devices]
    config_meta['device'] = device

    if not torch.cuda.is_available():
        placement = 'cpu'
        logger.info(f'path mode: {config.path_type}, placement: {placement}')
    else:
        if len(device) == 1:
            placement = 'single'
            torch.cuda.set_device(device[0])
        elif config_meta['auto_parallel']:
            placement = 'auto'
        else:
            placement = 'manual'

        logger.info(f'path mode: {config.path_type}, placement: {placement}, n_devices: {args.n_devices}')
    config_meta['placement'] = placement


def _place_model(model):
    # epoch, model, tokenizer, scores = load_checkpoint()
    if config_meta['placement'] == 'auto':
        model = nn.DataParallel(model, device_ids=config_meta['device'])
        logger.info(f'Parallel Data to devices: {config_meta["device"]}')

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

        # logger.info(f'pred: {pred}')
        # logger.info(f'pred[0]: {pred[0]}')
        if type(pred) is tuple:  # BertForSequenceClassification returns tuple
            pred = pred[0]

        n_cls = pred.size()[-1]
        if n_cls == 2:
            pred = F.softmax(pred, dim=-1)[:, 1]
        elif n_cls == 1:
            pred = pred.squeeze(-1)
        else:
            raise ValueError(f'Invalid n_cls: {n_cls}')

        rel_scores = pred.cpu().detach().numpy()  # d_batch,
        logger.info(f'rel_scores: {rel_scores.shape}')

        doc_rel_scores.append(rel_scores[:n_sents])

    rel_scores = np.concatenate(doc_rel_scores)

    dump_fp = join(dump_dp, cluster_loader.cid)

    tools.save_obj(obj=rel_scores, fp=dump_fp)
    logger.info(f'Dumping ranking file to: {dump_fp}')


def get_data_loader_gen(ir_rec_dp):
    loader_params = {
        'tokenize_narr': False,
        'query_type': rr_config.QUERY_TYPE,
        'retrieve_dp': ir_rec_dp,
        'with_sub': rr_config.WITH_SUB,
    }

    if config.meta_model_name in ('bert_rr'):
        loader_cls = QSDataLoader
    else:
        raise ValueError(f'Invalid meta_model_name: {config.meta_model_name}')

    data_gen = loader_cls(**loader_params)
    return data_gen


def dump_rel_scores():
    if exists(rel_scores_dp):
        raise ValueError(f'rel_scores_dp exists: {rel_scores_dp}')
    os.mkdir(rel_scores_dp)

    ir_rec_dp = join(path_parser.summary_rank, rr_config.IR_RECORDS_DIR_NAME)

    model = _place_model(model=config.bert_model)
    data_loader_generator = get_data_loader_gen(ir_rec_dp)
    for cluster_loader in data_loader_generator:
        if config.meta_model_name in ('bert_rr'):
            _dump(model, cluster_loader=cluster_loader, dump_dp=rel_scores_dp)
        else:
            raise ValueError(f'Invalid meta_model_name: {config.meta_model_name}')


def load_rel_scores(cid, rel_scores_dp):
    rel_scores_fp = join(rel_scores_dp, cid)
    return tools.load_obj(rel_scores_fp)


def rel_scores2rank():
    if exists(rank_dp):
        raise ValueError(f'rank_dp exists: {rank_dp}')
    os.mkdir(rank_dp)

    for cid in tqdm(cids):
        rel_scores = load_rel_scores(cid=cid, rel_scores_dp=rel_scores_dp)
        sent_ids = np.argsort(rel_scores)[::-1].tolist()

        sid_score_list = []
        for sid in sent_ids:
            sid_score = (f'0_{sid}', rel_scores[sid])
            sid_score_list.append(sid_score)

        original_sents, _ = load_retrieved_sentences(retrieved_dp=ir_rec_dp, cid=cid)
        rank_records = rank_sent.get_rank_records(sid_score_list, sents=original_sents)

        n_sents = rank_sent.dump_rank_records(rank_records=rank_records, out_fp=join(rank_dp, cid), with_rank_idx=False)
        logger.info(f'Dump {n_sents} ranking records')


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

    rr_tune_dp = join(path_parser.summary_rank, rr_config.RR_TUNE_DIR_NAME_BERT)
    rr_tune_result_fp = join(path_parser.tune, rr_config.RR_TUNE_DIR_NAME_BERT)
    with open(rr_tune_result_fp, mode='a', encoding='utf-8') as out_f:
        headline = 'Filter\tRecall\tF1\n'
        out_f.write(headline)

    for filter_var in tune_range:
        if exists(rr_tune_dp):  # remove previous output
            shutil.rmtree(rr_tune_dp)
        os.mkdir(rr_tune_dp)

        for cid in tqdm(cids):
            retrieval_params = {
                'model_name': rr_config.RR_MODEL_NAME_BERT,
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

        performance = rouge.compute_rouge_for_cont_sel_in_sentences(rr_tune_dp)
        with open(rr_tune_result_fp, mode='a', encoding='utf-8') as out_f:
            if rr_config.FILTER in ('conf', 'comp'):
                rec = '{0:.2f}\t{1}\n'.format(filter_var, performance)
            else:
                rec = f'{filter_var}\t{performance}\n'

            out_f.write(rec)

    if exists(rr_tune_dp):  # remove previous output
        shutil.rmtree(rr_tune_dp)
    print(f'Check tuning results at : {rr_tune_result_fp}')


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

    rr_tune_dp = path_parser.summary_rank / rr_config.RR_FINER_TUNE_DIR_NAME_BERT
    rr_tune_result_fp = path_parser.tune / f'{rr_config.RR_FINER_TUNE_DIR_NAME_BERT}.txt'
    with open(rr_tune_result_fp, mode='a', encoding='utf-8') as out_f:
        headline = 'Filter\tRecall\tF1\n'
        out_f.write(headline)

    for filter_var in tune_range:
        if exists(rr_tune_dp):  # remove previous output
            shutil.rmtree(rr_tune_dp)
        os.mkdir(rr_tune_dp)

        for cid in tqdm(cids):
            retrieval_params = {
                'model_name': rr_config.RR_MODEL_NAME_BERT,
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
                rec = f'{filter_var}\t{performance}\n'

            out_f.write(rec)

    if exists(rr_tune_dp):  # remove previous output
        shutil.rmtree(rr_tune_dp)


def _load_rank_items_joint_with_ir(rr_rank_dp, ir_record_dp, cid, interpolation, joint_scale_norm, rr_max_min_norm):
    rr_rank_fp = join(rr_rank_dp, cid)
    with io.open(rr_rank_fp, encoding='utf-8') as f:
        rr_rank = f.readlines()
    rank_items = [ll.rstrip('\n').split('\t') for ll in rr_rank]

    if rr_max_min_norm:
        rank_items, _ = ir_tools._norm(rank_items)

    # get rr_scores
    rr_sent_ids, rr_scores = [], []
    for items in rank_items:
        rr_sent_ids.append(int(items[0].split('_')[-1]))
        rr_scores.append(float(items[1]))
    rr_scores = np.array(rr_scores)
    # print('rr_sent_ids: {}'.format(rr_sent_ids))

    # get ir_scores
    ir_record_fp = join(ir_record_dp, cid)
    with io.open(ir_record_fp, encoding='utf-8') as f:
        ir_scores = [float(line.split('\t')[1]) for line in f.readlines()]
    ir_scores = np.array(ir_scores)
    ir_scores = ir_scores[rr_sent_ids]

    # interpolate
    if joint_scale_norm:
        rr_scores = rr_scores / np.sum(rr_scores)
        ir_scores = ir_scores / np.sum(ir_scores)
        print('normed: rr_scores: {}'.format(rr_scores))
        print('normed: ir_scores: {}'.format(ir_scores))

    joint_scores = interpolation * rr_scores + (1 - interpolation) * ir_scores
    joint_scores = joint_scores.tolist()

    # write into ranking items
    for item_idx in range(len(rank_items)):
        rank_items[item_idx][1] = joint_scores[item_idx]

    # re-rank as per new scores and revert to str
    rank_items = sorted(rank_items, key=lambda item: item[1], reverse=True)
    for items in rank_items:
        items[1] = str(items[1])

    return rank_items


def retrieve_joint_with_ir(rr_rank_dp,
                           ir_record_dp,
                           cid,
                           filter_var,
                           filter,
                           interpolation,
                           deduplicate,
                           min_ns,
                           joint_scale_norm,
                           rr_max_min_norm):
    rank_items = _load_rank_items_joint_with_ir(rr_rank_dp, ir_record_dp, cid, interpolation,
                                                joint_scale_norm, rr_max_min_norm)

    if filter == 'conf':
        retrieved_items = ir_tools._retrieve_from_rank_items_via_conf(rank_items,
                                                                      filter_var,
                                                                      deduplicate=deduplicate,
                                                                      min_ns=min_ns)
    elif filter == 'comp':
        retrieved_items = ir_tools._retrieve_from_rank_items_via_comp(rank_items,
                                                                      filter_var,
                                                                      deduplicate=deduplicate,
                                                                      min_ns=min_ns)
    else:
        raise ValueError(f'Invalid FILTER: {filter}')

    logger.info(f'retrieved {len(retrieved_items)}/{len(rank_items)} items for {cid}')

    return retrieved_items


def rr_rank2records():
    rr_rec_dp = join(path_parser.summary_rank, rr_config.RR_RECORD_DIR_NAME_BERT)

    if exists(rr_rec_dp):
        raise ValueError(f'rr_rec_dp exists: {rr_rec_dp}')
    os.mkdir(rr_rec_dp)

    for cid in tqdm(cids):
        retrieval_params = {
            'model_name': rr_config.RR_MODEL_NAME_BERT,
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


def rr_rank2records_in_batch():
    if rr_config.FILTER in ('conf', 'comp'):
        filter_var_range = np.arange(0.05, 1.05, 0.05)
    else:  # topK
        interval = 10
        if rr_config.ir_config.FILTER == 'topK':
            start = interval
            end = rr_config.ir_config.FILTER_VAR + interval
        else:
            start = 40
            end = 150 + interval
        filter_var_range = range(start, end, interval)

    for filter_var in tqdm(filter_var_range):
        rr_rec_dn = rr_config.RR_RECORD_DIR_NAME_PATTERN.format(rr_config.RR_MODEL_NAME_BERT, filter_var, rr_config.FILTER)
        rr_rec_dp = join(path_parser.summary_rank, rr_rec_dn)

        if exists(rr_rec_dp):
            raise ValueError(f'rr_rec_dp exists: {rr_rec_dp}')
        os.mkdir(rr_rec_dp)

        for cid in cids:
            retrieval_params = {
                'model_name': rr_config.RR_MODEL_NAME_BERT,
                'cid': cid,
                'filter_var': filter_var,
                'filter': rr_config.FILTER,
                'deduplicate': None,
                'min_ns': rr_config.RR_MIN_NS
            }

            # todo: examine if we want to normalize scores for bert_rr
            if meta_model_name.startswith('bert_squad') and config.squad_var.startswith('bert_shared'):
                retrieval_params['norm'] = True

            retrieved_items = ir_tools.retrieve(**retrieval_params)
            ir_tools.dump_retrieval(fp=join(rr_rec_dp, cid), retrieved_items=retrieved_items)


def rr_rank2records_with_ir_scores():
    assert rr_config.RR_INTERPOLATION

    rr_rec_dp = join(path_parser.summary_rank, rr_config.RR_RECORD_DIR_NAME_BERT)

    if exists(rr_rec_dp):
        raise ValueError(f'rr_rec_dp exists: {rr_rec_dp}')
    os.mkdir(rr_rec_dp)

    for cid in tqdm(cids):
        retrieval_params = {
            'rr_rank_dp': rank_dp,
            'ir_record_dp': ir_rec_dp,
            'cid': cid,
            'filter_var': rr_config.FILTER_VAR,
            'filter': rr_config.FILTER,
            'deduplicate': None,
            'min_ns': rr_config.RR_MIN_NS,
            'interpolation': rr_config.RR_INTERPOLATION,
            'joint_scale_norm': rr_config.RR_INTERPOLATION_NORM,
            'rr_max_min_norm': True if config.meta_model_name == 'bert_squad' else False,
        }

        retrieved_items = retrieve_joint_with_ir(**retrieval_params)
        ir_tools.dump_retrieval(fp=join(rr_rec_dp, cid), retrieved_items=retrieved_items)


def line_counter():
    rr_rec_dp = join(path_parser.summary_rank, rr_config.RR_RECORD_DIR_NAME_BERT)

    if not exists(rr_rec_dp):
        raise ValueError(f'rr_rec_dp does not exist: {rr_rec_dp}')

    logger.info(f'rr_rec_dp: {rr_rec_dp}')

    doc_fps = [join(rr_rec_dp, fn) for fn in listdir(rr_rec_dp)
               if isfile(join(rr_rec_dp, fn)) and not join(rr_rec_dp, fn).endswith('.swp')]

    n_sents = []
    for doc_fp in doc_fps:
        with io.open(doc_fp, encoding='utf-8') as f:
            n_sents.append(len(f.readlines()))

    logger.info(f'n_sents: {n_sents}')


def select_e2e():
    """
        This function is for ablation study (w/o Centrality).

    """
    ir_rec_dp = join(path_parser.summary_rank, rr_config.IR_RECORDS_DIR_NAME)
    params = {
        'model_name': rr_config.RR_MODEL_NAME_BERT,
        'cos_threshold': 0.6,
        'retrieved_dp': ir_rec_dp,
        'max_n_summary_words': 1000,
    }

    select_sent.select_end2end(**params)


def compute_rouge():
    text_params = {
        'model_name': rr_config.RR_MODEL_NAME_BERT,
        'cos_threshold': 0.6,
    }
    output = rouge.compute_rouge(**text_params)


def get_text_dp():
    """
        Copied from bert_marge/main.py.
        
    """

    assert rr_config.USE_TEXT
    dn = rr_config.TEXT_DIR_NAME
    text_dp = path_parser.summary_text / dn
    print(f'Build from text_dp: {text_dp}')
    return text_dp


def eval_unilm_out():
    unilm_eval = UniLMEval(marge_config=rr_config, 
        pre_tokenize_sent=False, 
        max_eval_len=250, 
        cluster_ids=cids,
        eval_tdqfs=False)
    perf = unilm_eval.build_and_eval_unilm_out()
    print(perf)
    return perf
    # unilm_eval.eval_unilm_out()


def eval_in_batch(unilm_model_id, start=500, end=10000, intv=5000, unilm_ckpts=None):
    if not unilm_ckpts:
        unilm_ckpts = range(start, end, intv)
    records = []
    for ckpt in unilm_ckpts:
        rr_config.override_glob_vars(unilm_model_id, ckpt)
        perf = eval_unilm_out()
        rec = f'{ckpt}: {perf}'
        records.append(rec)
    print('\n'.join(records))
    

def build_unilm_input(src):
    """
        Copied from bert_marge/main.py.
        
    """
    if src == 'rank':
        rank_dp = join(path_parser.summary_rank, rr_config.RR_RANK_DIR_NAME_BERT)
        text_dp = None
    elif src == 'text':
        rank_dp = None
        text_dp = get_text_dp()
    
    unilm_in_params = {
        'marge_config': rr_config,
        'rank_dp': rank_dp,
        'text_dp': text_dp,
        'fix_input': True,
        'cluster_ids': cids,
        'prepend_len': rr_config.PREPEND_LEN,
        'prepend_query': rr_config.PREPEND_QUERY,
    }
    unilm_input = UniLMInput(**unilm_in_params)
    
    if src == 'rank':
        unilm_input.build_from_rank()
    elif src == 'text':
        unilm_input.build_from_text()


if __name__ == '__main__':
    # init()
    # dump_rel_scores()
    # rel_scores2rank()

    # tune()  # select div hyper-parameter
    # finer_tune()  # visualize n_sents

    # build_unilm_input(src='rank')
    # eval_unilm_out()
    # unilm_ckpts=[3000]
    # eval_in_batch(unilm_model_id=26, start=1500, end=20000, intv=1500, unilm_ckpts=unilm_ckpts)
    
    # rr_rank2records()  # with selected hyper-parameter
    # select_e2e()
    compute_rouge()
