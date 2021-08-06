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
import json
from sklearn.metrics.pairwise import cosine_similarity
import shutil

import summ.rank_sent as rank_sent
import summ.select_sent as select_sent
import summ.compute_rouge as rouge
import ir.ir_tools as ir_tools
from bert_rr.data_pipe_mn_train_val import (QSDataLoader, get_test_cc_ids, load_cluster)
import bert_rr.rr_config as rr_config

from unilm_utils.unilm_eval import UniLMEval
from unilm_utils.unilm_input import UniLMInput


"""
    Function: build training/dev datasets for UniLM
    Method: 
        1. Use trained RR model to score sentences, and dump the scores
        2. Load sentence scores and build rank files from dum
        3. Build UniLM input files based on the rank files, i.e., topK.
        
    This module builds the following pipeline:

    rel_scores [compute, dump] =>
    rr_rank [load rel_scores, compute, dump]=>
    rr_records [load rr_rank, compute, dump]

"""


# specify this
DATASET_VAR = 'train'
assert DATASET_VAR in ('train', 'val'), f'Invalid dataset_var: {DATASET_VAR}'

# input
MARGE_CLUSTER_NAME = 'marge_cluster-ratio-reveal_0.0'
dp_marge_cluster = path_parser.data / 'multinews' / MARGE_CLUSTER_NAME / DATASET_VAR

# output
SCORE_NAME = f'{rr_config.RELEVANCE_SCORE_DIR_NAME_MN}-{MARGE_CLUSTER_NAME}-{DATASET_VAR}'
rel_scores_dp = path_parser.graph_rel_scores / SCORE_NAME

ESTIMATED_ROUGE_DIR_NAME = f'rouge_estimated-{rr_config.RELEVANCE_SCORE_DIR_NAME_MN}-{MARGE_CLUSTER_NAME}'
estimated_rouge_dir = path_parser.data / 'multinews' / ESTIMATED_ROUGE_DIR_NAME


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
    data_gen = QSDataLoader(retrieve_dp=dp_marge_cluster, 
        no_query=rr_config.NO_QUERY,
        with_sub=rr_config.WITH_SUB)

    for cluster_loader in data_gen:
        if config.meta_model_name in ('bert_rr'):
            _dump(model, cluster_loader=cluster_loader, dump_dp=rel_scores_dp)
        else:
            raise ValueError('Invalid meta_model_name: {}'.format(config.meta_model_name))


def load_rel_scores(cid, rel_scores_dp):
    rel_scores_fp = join(rel_scores_dp, cid)
    return tools.load_obj(rel_scores_fp)


def rel_scores2estimate_rouge_file():
    """
        Convert rel_scores into a file with sentences and their estimated ROUGE.
    """
    cids = get_test_cc_ids(rel_scores_dp)
    if not exists(estimated_rouge_dir):
        os.mkdir(estimated_rouge_dir)
    
    out_fp = estimated_rouge_dir / f'{DATASET_VAR}.json'
    if exists(out_fp):
        raise ValueError(f'estimated_rouge_dir exists: {out_fp}')

    for cid in tqdm(cids):
        rel_scores = load_rel_scores(cid=cid, rel_scores_dp=rel_scores_dp)
        sent_indices = np.argsort(rel_scores)[::-1].tolist()

        _, sids, original_sents, _ = load_cluster(retrieved_dp=dp_marge_cluster, cid=cid)
        records = []

        for sent_idx in sent_indices:
            json_obj = {
                'sid': sids[sent_idx],
                'sent': original_sents[0][sent_idx],
                'estimated_rouge': str(rel_scores[sent_idx]),
            }
            records.append(json.dumps(json_obj, ensure_ascii=False))
        
        with open(out_fp, mode='a', encoding='utf-8') as f:
            f.write('\n'.join(records)+'\n')

        logger.info(f'Dump {len(records)} ranking records')


if __name__ == '__main__':
    # init()
    # dump_rel_scores()
    rel_scores2estimate_rouge_file()
