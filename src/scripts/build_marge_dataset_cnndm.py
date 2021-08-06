# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
import logging
from summ import compute_rouge
from data.mn_parser import MultiNewsParser
import json

from pyrouge import Rouge155
import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import utils.tools as tools
from tqdm import tqdm
import shutil
from multiprocessing import Pool

import io
import numpy as np

"""
    This script builds train/val/test datasets for MaRGE training and dev.

    You have to calculate ROUGE scores for all passages/sentences in CNNDM data using:
        - dump_passage_rouge_mp.py, or
        - dump_sentence_rouge_mp.py

    Specify DATASET_VAR before running.
    
"""

if config.path_type == 'afs':
    DP_PROJ = Path('/disk/scratch/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_DATA = DP_PROJ / 'data'
DP_CNNDM = DP_DATA / 'cnndm'

DATASET_VARS = ['train', 'val', 'test']

# specify this
DATASET_VAR = 'train'
assert DATASET_VAR in DATASET_VARS, f'Invalid dataset_var: {DATASET_VAR}'

FP_ROUGE_CNNDM = DP_CNNDM / 'rouge' / f'{DATASET_VAR}.json'
# FN_MASKED_CNNDM_SUMMARY = f'{DATASET_VAR}-max-max_reveal_0.5.json'
FN_MASKED_CNNDM_SUMMARY = f'{DATASET_VAR}-ratio-reveal_0.0.json'
FP_MASKED_CNNDM_SUMMARY = DP_CNNDM / 'masked_cnndm_summary' / FN_MASKED_CNNDM_SUMMARY

if not exists(FP_ROUGE_CNNDM):
    raise ValueError(f'FP_ROUGE_CNNDM does not exists: {FP_ROUGE_CNNDM}. \
        Calculate ROUGE for CNNDM segments before constructing MaRGE datasets.')

if not exists(FP_MASKED_CNNDM_SUMMARY):
    raise ValueError(f'FP_MASKED_CNNDM_SUMMARY does not exists: {FP_MASKED_CNNDM_SUMMARY}. \
        Mask CNNDM summaries before constructing MaRGE datasets.')

USE_MINI_L3 = False
NUM_NEG = 3
FN_MARGE_META = 'marge' if not USE_MINI_L3 else f'marge_l3{NUM_NEG}'

FN_MARGE = f"{FN_MARGE_META}-{FN_MASKED_CNNDM_SUMMARY[:-5]}"  # remove .json

USE_INTER_SENT_SEP = False
if USE_INTER_SENT_SEP:
    FN_MARGE += '-sent_sep'
FP_MARGE = DP_CNNDM / 'marge' / f'{FN_MARGE}.json'
# os.mkdir(FP_MARGE)

def get_cid2summary(use_intersent_sep):
    cid = 0
    cid2summary = {}
    with open(FP_MASKED_CNNDM_SUMMARY) as masked_summary_f:
        for line in masked_summary_f:
            key = 'masked_seq_with_sep' if use_intersent_sep else 'masked_seq'
            masked_seq = json.loads(line)[key]
            cid2summary[cid] = masked_seq
            cid += 1
    
    return cid2summary
    

def _get_cid(json_obj):
    return int(json_obj['sid'].split('_')[0])


def build(use_intersent_sep):
    if exists(FP_MARGE):
        raise ValueError(f'FP_MARGE already exists: {FP_MARGE}')

    cid2summary = get_cid2summary(use_intersent_sep)

    with open(FP_ROUGE_CNNDM) as rouge_mn_f:
        lines = rouge_mn_f.readlines()
        with open(FP_MARGE, 'a') as dump_f:
            for line in tqdm(lines):
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)
                json_obj['masked_summary'] = cid2summary[_cid]
                json_str = json.dumps(json_obj, ensure_ascii=False)
                dump_f.write(f'{json_str}\n')
    logger.info(f'Sucessfully dump {DATASET_VAR} set to: {FP_MARGE}')


def _proc_mini_l3(doc_objs, summary, num_neg=3):
    """
        Get the first 3 objs and sample #num_neg objs from the rest.

    """
    if len(doc_objs) < num_neg+3:
        return None

    import copy
    import random
    mini_objs = copy.deepcopy(doc_objs[:3])

    neg_indices = sorted(random.sample(list(range(3, len(doc_objs))), num_neg))
    for nid in neg_indices:
        mini_objs.append(doc_objs[nid])

    assert len(mini_objs) == num_neg+3

    json_str = ''
    for json_obj in mini_objs:
        json_obj['masked_summary'] = summary
        _j_str = json.dumps(json_obj, ensure_ascii=False)
        json_str += f'{_j_str}\n'
    return json_str


def build_mini_l3(use_intersent_sep, num_neg):
    if exists(FP_MARGE):
        raise ValueError(f'FP_MARGE already exists: {FP_MARGE}')

    cid2summary = get_cid2summary(use_intersent_sep)

    with open(FP_ROUGE_CNNDM) as rouge_mn_f:
        lines = rouge_mn_f.readlines()
        with open(FP_MARGE, 'a') as dump_f:
            doc_objs = []
            cur_cid = None

            for line in tqdm(lines):
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)

                if not cur_cid or cur_cid == _cid:
                    cur_cid = _cid
                    doc_objs.append(json_obj)
                else:
                    json_str = _proc_mini_l3(doc_objs, 
                        summary=cid2summary[cur_cid], num_neg=num_neg)
                    if json_str:
                        dump_f.write(json_str)
                    doc_objs = [json_obj]
                    cur_cid = _cid
    
    logger.info(f'Sucessfully dump {DATASET_VAR} set to: {FP_MARGE}')


def _draw(scores, range, n_bins, xlabel, title, color='darkblue'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    scores = [float(sc) for sc in scores]
    counts, bin_edges = np.histogram(scores, bins=n_bins, range=range, density=False)
    dist = [c/float(sum(counts)) for c in counts]

    logger.info(f'total counts: {sum(counts)}')
    logger.info(f'distribution: {dist}')
    logger.info(f'bin_edges: {bin_edges}')

    fig = plt.figure(figsize=(5, 4))
    sns.distplot(scores, hist=True, kde=True, 
        bins=n_bins, 
        color=color, 
        hist_kws={'edgecolor':'black', 'range': range}, 
        kde_kws={'linewidth': 2})
    # plt.hist(np.array(n_words), bins=7, range=(10, 80), density=True, stacked=True)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.tight_layout()
    
    fig.savefig(DP_PROJ/'stats'/ title, bbox_inches='tight')
    # plt.show()


def rouge_stats():
    """
        Draw and dump the distributions of ROUGE-2 Recall/F1 over sentences.
    
    """
    fp_rouge_stats = DP_PROJ / 'stats' / f'rouge.txt'

    if not exists(fp_rouge_stats):
        logger.info('Build stat file')
        with io.open(fp_rouge_stats, mode='a') as stat_f:
            stat_f.write('rouge_2_recall\trouge_2_f1\n')
            with io.open(FP_ROUGE_CNNDM) as mask_f:
                lines = mask_f.readlines()
                for line in tqdm(lines):
                    line = line.strip('\n')
                    if not line:
                        continue
                    json_dict = json.loads(line)
                    stat_f.write(f"{json_dict['rouge_2_recall']}\t{json_dict['rouge_2_f1']}\n")
    
    logger.info(f'Read from stat file: {fp_rouge_stats}')
    with io.open(fp_rouge_stats) as stat_f:
        lines = stat_f.readlines()[1:]
        items = [line.strip('\n').split('\t') for line in lines]
        rouge_2_recall, rouge_2_f1 = zip(*items)
        _draw(rouge_2_recall, range=[0.0, 0.3], n_bins=30, color='darkblue',
            xlabel='ROUGE-2 Recall', title='rouge_2_recall.pdf')
        _draw(rouge_2_f1, range=[0.0, 0.3], n_bins=30, color='darkred',
            xlabel='ROUGE-2 F1', title='rouge_2_f1.pdf')


def rouge_zero_nonzero():
    fp_rouge_stats = DP_PROJ / 'stats' / f'rouge.txt'
    if not exists(fp_rouge_stats):
        logger.info('Build stat file')
        with io.open(fp_rouge_stats, mode='a') as stat_f:
            stat_f.write('rouge_2_recall\trouge_2_f1\n')
            with io.open(FP_ROUGE_CNNDM) as mask_f:
                lines = mask_f.readlines()
                for line in tqdm(lines):
                    line = line.strip('\n')
                    if not line:
                        continue
                    json_dict = json.loads(line)
                    stat_f.write(f"{json_dict['rouge_2_recall']}\t{json_dict['rouge_2_f1']}\n")
    
    logger.info(f'Read from stat file: {fp_rouge_stats}')
    with io.open(fp_rouge_stats) as stat_f:
        lines = stat_f.readlines()[1:]
        items = [line.strip('\n').split('\t') for line in lines]
        rouge_2_recall, rouge_2_f1 = zip(*items)
        rouge_2_recall = [float(sc) for sc in rouge_2_recall]
        rouge_2_f1 = [float(sc) for sc in rouge_2_f1]
        
        # recall: 0: 6987199, >0: 4010079
        # f1: 0: 6987199, >0: 4010079
        # recall_zero = rouge_2_recall.count(0.0)
        # f1_zero = rouge_2_f1.count(0.0)
        # logger.info(f'ROUGE-2 Recall: zero: {recall_zero}, non-zero: {len(items)-recall_zero}')
        # logger.info(f'ROUGE-2 F1: zero: {f1_zero}, non-zero: {len(items)-f1_zero}')

        # thre=0.01
        # recall: 0: 7057889, >0: 3939389
        # f1: 0: 6989832, >0: 4007446
        thre = 0.05
        recall_tiny = len([sc for sc in rouge_2_recall if sc < thre])
        f1_tiny = len([sc for sc in rouge_2_f1 if sc < thre])
        
        logger.info(f'ROUGE-2 Recall: tiny: {recall_tiny}, non-tiny: {len(items)-recall_tiny}')
        logger.info(f'ROUGE-2 F1: tiny: {f1_tiny}, non-tiny: {len(items)-f1_tiny}')

        # thre=0.5
        # recall: 0: 9497485, >0: 1499793
        # f1: 0: 9045873, >0: 1951405


if __name__ == "__main__":
    # build(use_intersent_sep=USE_INTER_SENT_SEP)
    build_mini_l3(use_intersent_sep=USE_INTER_SENT_SEP, num_neg=NUM_NEG)
    # rouge_stats()
    # rouge_zero_nonzero()
