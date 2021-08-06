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

"""
    This script builds train/val/test datasets for training summarization model on MultiNews. 

    Before using this script, you should build train/val/test datasets for ROUGE Regression first.

    This script ranks all sentences/passages (determined by N_SENTS) according to a metric (e.g., ROUGE Recall or F1). 

    During training Summarization model, you can take the TopK of them by setting args.max_n_passages. 
"""

if config.path_type == 'afs':
    DP_PROJ = Path('/disk/scratch/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_DATA = DP_PROJ / 'data'
DATASET_VARS = ['train', 'val', 'test']
MIN_NW_MODES = ['max', 'sample']  # for masked summary file names

## Specify the following configs
N_SENTS = 1  # 1, 4, 8
MIN_NW_MODE = 'sample'
# N_TOP = 90
## Specify the above configs

if N_SENTS == 1:
    RR_ID_KEY = 'sid'
    RR_PASSAGE_KEY = 'sentence'
else:
    RR_ID_KEY = 'pid'
    RR_PASSAGE_KEY = 'passage'

if MIN_NW_MODE not in MIN_NW_MODES:
    raise ValueError(f'Invalid min_nw_mode: {min_nw_mode}')

DP_MASKED_SUMMARY = DP_DATA / 'masked_mn_summary' / MIN_NW_MODE
DP_ROUGE_REGRESSION = DP_DATA / 'rouge_regression' / f'rr_{MIN_NW_MODE}_{N_SENTS}'

SELECTION_METRIC = 'rouge_2_f1'  # rouge_2_recall, rouge_2_f1

DP_TOP = DP_DATA / 'top_mn' / f'top_mn_{MIN_NW_MODE}_{N_SENTS}_{SELECTION_METRIC.split("_")[-1]}'

if not exists(DP_MASKED_SUMMARY):
    raise ValueError(f'DP_ROUGE_MN does not exists: {DP_MASKED_SUMMARY}. MASK MN summaries before constructing Top MN datasets.')

if not exists(DP_ROUGE_REGRESSION):
    raise ValueError(f'DP_ROUGE_REGRESSION does not exists: {DP_ROUGE_REGRESSION}. Construct RR datasets before constructing Top MN datasets.')

if exists(DP_TOP):
    raise ValueError(f'DP_TOP already exists: {DP_TOP}')
os.mkdir(DP_TOP)


def get_cid2summary(dataset_var):
    if dataset_var == 'train_debug':  # partial training set for model dev
        dataset_var = 'train'
        
    masked_summary_fp = DP_MASKED_SUMMARY / f'{dataset_var}.json'
    cid = 0
    cid2summary = {}
    with open(masked_summary_fp) as masked_summary_f:
        for line in masked_summary_f:
            json_obj = json.loads(line)
            cid2summary[cid] = {
                'masked_seq': json_obj['masked_seq'],
                'original_summary': json_obj['original_summary'],
            }
            cid += 1
    
    return cid2summary
    

def _get_cid(json_obj):
    return int(json_obj[RR_ID_KEY].split('_')[0])


def _rank_passage_objs(passage_objs, metric):
    return sorted(passage_objs, key=lambda po: po[metric], reverse=True)


def build(dataset_var):
    if dataset_var not in DATASET_VARS:
        raise ValueError(f'Invalid dataset_var: {dataset_var}')

    rr_fp = DP_ROUGE_REGRESSION / f'{dataset_var}.json'
    dump_fp = DP_TOP / f'{dataset_var}.json'
    cid2summary = get_cid2summary(dataset_var)

    cid = 0
    passages_objs = []
    with open(dump_fp, 'a') as dump_f:
        with open(rr_fp) as rr_f:
            for line in rr_f:
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)
                if _cid != cid:
                    ranked_passages_objs = _rank_passage_objs(passages_objs, metric=SELECTION_METRIC)

                    if cid % 500 == 0:
                        logger.info(f'cid: {cid}, #Passages: {len(passages_objs)}')
                    
                    cluster_obj = {
                        'cid': cid,
                        'passages': ranked_passages_objs,
                        **cid2summary[cid],  # this include masked and original summary
                    }
                    json_str = json.dumps(cluster_obj, ensure_ascii=False)
                    dump_f.write(f'{json_str}\n')

                    passages_objs = []

                po = {
                    'id': json_obj[RR_ID_KEY],
                    'passage': json_obj[RR_PASSAGE_KEY],
                    'rouge_2_recall': json_obj['rouge_2_recall'],
                    'rouge_2_f1': json_obj['rouge_2_f1'],
                }
                passages_objs.append(po)
                cid = _cid
    
    logger.info(f'Sucessfully dump {dataset_var} set to: {dump_fp}!')


def build_all():
    for dataset_var in DATASET_VARS:
        build(dataset_var)
    

if __name__ == "__main__":
    # dataset_var = 'test'
    # build(dataset_var=dataset_var)
    build_all()
