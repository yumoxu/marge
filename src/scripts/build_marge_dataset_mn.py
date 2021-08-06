# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
import logging
import itertools
import json
import copy
import random
from tqdm import tqdm
import shutil
from multiprocessing import Pool
from pyrouge import Rouge155

from summ import compute_rouge
from data.mn_parser import MultiNewsParser
import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import utils.tools as tools


"""
    This script builds train/val/test datasets for MaRGE on MultiNews.

    Before using this script, you have to:
    - calculate ROUGE scores for all sentences from MultiNews clusters
    - mask summaryies from MultiNews

    Specify the following vars before running:
    - DATASET_VAR
    - FN_MASKED_MN_SUMMARY
    - USE_MINI_DATA, NUM_POS, NUM_NEG
    - USE_INTER_SENT_SEP

"""

if config.path_type == 'afs':
    DP_PROJ = Path('/disk/scratch/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_DATA = DP_PROJ / 'data'
DP_MN = DP_DATA / 'multinews'

DATASET_VARS = ['train', 'val', 'test']

# specify this
DATASET_VAR = 'train'
assert DATASET_VAR in DATASET_VARS, f'Invalid dataset_var: {DATASET_VAR}'

FP_ROUGE_MN = DP_MN / 'rouge' / f'{DATASET_VAR}.json'
if not exists(FP_ROUGE_MN):
    raise ValueError(f'FP_ROUGE_MN does not exists: {FP_ROUGE_MN}. \
        Calculate ROUGE for MN segments before constructing MaRGE datasets.')

# -ratio-reveal_0.0.json'
# -rand-reveal_n_abs_non.json
# -rand-reveal_0.85.json
FN_MASKED_MN_SUMMARY = f'{DATASET_VAR}-ratio-reveal_0.9.json'

FP_MASKED_MN_SUMMARY = DP_MN / 'masked_mn_summary' / FN_MASKED_MN_SUMMARY
if not exists(FP_MASKED_MN_SUMMARY):
    raise ValueError(f'FP_MASKED_MN_SUMMARY does not exists: {FP_MASKED_MN_SUMMARY}. \
        Mask MN summaries before constructing MaRGE datasets.')

USE_MINI_DATA = True
NUM_POS = 20
NUM_NEG = 20
FN_MARGE_META = 'marge' if not USE_MINI_DATA else f'marge_l{NUM_POS}{NUM_NEG}'

RATIO_IS_ONE = False  # build dataset for the NO-MASK baseline
if RATIO_IS_ONE:
    FN_MARGE = f"{FN_MARGE_META}-{DATASET_VAR}-ratio-reveal_1.0"
else:
    FN_MARGE = f"{FN_MARGE_META}-{FN_MASKED_MN_SUMMARY[:-5]}"  # remove .json

MASKED_SUMMARY_KEY = 'masked_seq'  # masked_seq_with_sub
if MASKED_SUMMARY_KEY == 'masked_seq_with_sub':
    FN_MARGE += '-with_sub'
elif MASKED_SUMMARY_KEY == 'masked_seq_with_sep':
    FN_MARGE += '-with_sep'

FP_MARGE = DP_MN / 'marge' / f'{FN_MARGE}.json'


def get_cid2summary():
    cid = 0
    cid2summary = {}
    with open(FP_MASKED_MN_SUMMARY) as masked_summary_f:
        for line in masked_summary_f:
            if RATIO_IS_ONE:
                summ_seq = json.loads(line)['original_summary']
                # sentences = json.loads(line)['sentences']
                # words = [sent['words'] for sent in sentences]
                # import itertools
                # summ_seq = list(itertools.chain(*words))
            else:
                summ_seq = json.loads(line)[MASKED_SUMMARY_KEY]
            
            cid2summary[cid] = summ_seq
            cid += 1
    
    return cid2summary
    

def _get_cid(json_obj):
    return int(json_obj['sid'].split('_')[0])

def _get_did(json_obj):
    return int(json_obj['sid'].split('_')[1])


def build():
    if exists(FP_MARGE):
        raise ValueError(f'FP_MARGE already exists: {FP_MARGE}')

    cid2summary = get_cid2summary()

    with open(FP_ROUGE_MN) as rouge_mn_f:
        lines = rouge_mn_f.readlines()
        with open(FP_MARGE, 'a') as dump_f:
            for line in tqdm(lines):
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)
                json_obj['sentence'] = json_obj['sentence'].replace('NEWLINE_CHAR', '').strip()
                json_obj['masked_summary'] = cid2summary[_cid]
                json_str = json.dumps(json_obj, ensure_ascii=False)
                dump_f.write(f'{json_str}\n')
    logger.info(f'Sucessfully dump {DATASET_VAR} set to: {FP_MARGE}')


def _proc_mini(cluster_objs, summary, num_pos=20, num_neg=20):
    """
        Get the first num_pos/N objs from each doc as positive samples,
        and sample #num_neg objs from the rest as negative samples.

    """
    def _select_objs():
        if len(cluster_objs) <= num_pos + num_neg:
            return cluster_objs

        # build new_cluster_objs
        cur_did = 0
        doc_objs = []
        new_cluster_objs = []  # organized via doc
        for json_obj in cluster_objs:
            _did = _get_did(json_obj)
            if cur_did == _did:
                doc_objs.append(json_obj)
            else:
                new_cluster_objs.append(doc_objs)
                doc_objs = [json_obj]
                cur_did = _did
        new_cluster_objs.append(doc_objs)
        
        num_objs = sum([len(doc_objs) for doc_objs in new_cluster_objs])
        assert num_objs == len(cluster_objs), f'{num_objs}, {len(cluster_objs)}'
        
        num_doc = len(new_cluster_objs)
        pos_objs = [[] for i in range(num_doc)]
        
        n_pos = 0
        _cluster_objs = copy.deepcopy(new_cluster_objs)
        while True:
            for idx, doc_objs in enumerate(_cluster_objs):
                if n_pos == num_pos:
                    break

                if not doc_objs:
                    continue
                
                pos_objs[idx].append(doc_objs[0])
                n_pos += 1
                # print(f'n_pos: {n_pos}')
                # print(f'pos_objs: {pos_objs}')
                _cluster_objs[idx] = doc_objs[1:]
            
            if n_pos == num_pos:
                break
        
        # print(f'pos_objs: {pos_objs}')
        pos_ends = [len(doc_pos) for doc_pos in pos_objs]
        # print(f'pos_ends: {pos_ends}')
        
        neg_pool = [doc_objs[pos_ends[idx]:] for idx, doc_objs in enumerate(new_cluster_objs)]
        neg_pool = list(itertools.chain(*neg_pool))
        # print(f'neg_pool: {len(neg_pool)}, num_neg: {num_neg}')
        neg_objs = random.sample(neg_pool, num_neg)
        pos_objs = list(itertools.chain(*pos_objs))
        assert len(pos_objs) == num_pos, f'pos_objs: {pos_objs}'
        assert len(neg_objs) == num_neg, f'neg_objs: {neg_objs}'
        # for doc_objs in new_cluster_objs:
        #     ns = min(len(doc_objs), avg_ns_per_doc)
        #     pos_objs.append(doc_objs[:ns])
            
        #     if ns == len(doc_objs):
        #         continue

        #     neg_pool = list(range(ns, len(doc_objs)))
        #     if avg_ns_per_doc >= len(neg_pool):
        #         neg_indices = neg_pool
        #     else:
        #         neg_indices = sorted(random.sample(neg_pool, avg_ns_per_doc))
            
        #     doc_neg_objs = [doc_objs[nid] for nid in neg_indices]
        #     pos_objs.append(doc_neg_objs)
        
        mini_objs = pos_objs + neg_objs
        return mini_objs
        
    mini_objs = _select_objs()
    json_str = ''
    for json_obj in mini_objs:
        json_obj['sentence'] = json_obj['sentence'].replace('NEWLINE_CHAR', '').strip()
        json_obj['masked_summary'] = summary
        _j_str = json.dumps(json_obj, ensure_ascii=False)
        json_str += f'{_j_str}\n'
    return json_str


def build_mini_data(num_pos, num_neg):
    """
        This is the func we used to generate the data which gives us the best results.

        However, there is a small bug about the first and second clusters; 
        they were mapped to the same summary.

    """
    if exists(FP_MARGE):
        raise ValueError(f'FP_MARGE already exists: {FP_MARGE}')

    cid2summary = get_cid2summary()

    with open(FP_ROUGE_MN) as rouge_mn_f:
        lines = rouge_mn_f.readlines()
        with open(FP_MARGE, 'a') as dump_f:
            cluster_objs = []
            cur_cid = None

            for line in tqdm(lines):
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)

                if not cur_cid or cur_cid == _cid:
                    cur_cid = _cid
                    cluster_objs.append(json_obj)
                else:
                    json_str = _proc_mini(cluster_objs, summary=cid2summary[cur_cid], num_neg=num_neg)
                    if json_str:
                        dump_f.write(json_str)
                    cluster_objs = [json_obj]
                    cur_cid = _cid
            
            json_str = _proc_mini(cluster_objs, summary=cid2summary[cur_cid], num_neg=num_neg)
            if json_str:
                dump_f.write(json_str)
    
    logger.info(f'Sucessfully dump {DATASET_VAR} set to: {FP_MARGE}')


def build_mini_data_with_zero_fix(num_pos, num_neg):
    if exists(FP_MARGE):
        raise ValueError(f'FP_MARGE already exists: {FP_MARGE}')

    cid2summary = get_cid2summary()

    with open(FP_ROUGE_MN) as rouge_mn_f:
        lines = rouge_mn_f.readlines()
        with open(FP_MARGE, 'a') as dump_f:
            cluster_objs = []
            cur_cid = 0

            for line in tqdm(lines):
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)

                # if not cur_cid or cur_cid == _cid:
                if cur_cid == _cid:  # TODO fix this condition in other building files
                    cluster_objs.append(json_obj)
                else:
                    json_str = _proc_mini(cluster_objs, summary=cid2summary[cur_cid], num_neg=num_neg)
                    if json_str:
                        dump_f.write(json_str)
                    cluster_objs = [json_obj]
                    cur_cid = _cid
            
            json_str = _proc_mini(cluster_objs, summary=cid2summary[cur_cid], num_neg=num_neg)
            if json_str:
                dump_f.write(json_str)
    
    logger.info(f'Sucessfully dump {DATASET_VAR} set to: {FP_MARGE}')


if __name__ == "__main__":
    if USE_MINI_DATA:
        build_mini_data(num_pos=NUM_POS, num_neg=NUM_NEG)
    else:
        build()
