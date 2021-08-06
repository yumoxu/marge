# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import json
from tqdm import tqdm
import io
from data.mn_parser import MultiNewsParser


"""
    This module turn MN data into DUC format, i.e., organize sentences and summaries via Cluster. 
    
    Input: a MaRGE json file
    Output: a folder for each cluster in the input

    For training, 44,964/44,972 clusters are converted.
    The following clusters do not exist in the sentence ROUGE file since they are empty
        453, 16290, 16489, 18812, 19279, 21620, 30735, 41993.
    These clusers are ignored in this conversion as well.
    
    The original indices for clusters are kept, to keep the mapping with their summaries.

"""

if config.path_type == 'afs':
    DP_PROJ = Path('/disk/nfs/ostrom/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_MN = DP_PROJ / 'data' / 'multinews'

DATASET_VARS = ['train', 'val', 'test']
# specify this
DATASET_VAR = 'train'
assert DATASET_VAR in DATASET_VARS, f'Invalid dataset_var: {DATASET_VAR}'

MARGE_DATASET_NAME = 'marge-ratio-reveal_0.0'
MARGE_CLUSTER_NAME = MARGE_DATASET_NAME.replace('marge', 'marge_cluster')

FP_MARGE_DATASET =  DP_MN / MARGE_DATASET_NAME / f'{DATASET_VAR}.json'

DP_MARGE_CLUSTER = DP_MN / MARGE_CLUSTER_NAME
if not exists(DP_MARGE_CLUSTER):
    os.mkdir(DP_MARGE_CLUSTER)

DP_MARGE_CLUSTER = DP_MARGE_CLUSTER / DATASET_VAR
assert not exists(DP_MARGE_CLUSTER), f'DP_MARGE_CLUSTER already exists: {DP_MARGE_CLUSTER}'
os.mkdir(DP_MARGE_CLUSTER)


def _get_cid(json_obj):
    return int(json_obj['sid'].split('_')[0])


def _get_did(json_obj):
    return int(json_obj['sid'].split('_')[1])


def dump(cluster_objs, fp):
    json_str = ''
    for json_obj in cluster_objs:
        _j_str = json.dumps(json_obj, ensure_ascii=False)
        json_str += f'{_j_str}\n'

    assert json_str, f'Cannot dump empty content to fp: {fp}'
    io.open(fp, 'a').write(json_str)


def convert():
    cluster_objs = []
    cur_cid = 0

    lines = io.open(FP_MARGE_DATASET).readlines()
    for line in tqdm(lines):
        line = line.strip('\n')
        if not line:
            continue
        json_obj = json.loads(line)
        _cid =  _get_cid(json_obj)

        if cur_cid == _cid:
            cluster_objs.append(json_obj)
        else:
            dump(cluster_objs, fp=DP_MARGE_CLUSTER/str(cur_cid))
            cluster_objs = [json_obj]
            cur_cid = _cid
        
    dump(cluster_objs, fp=DP_MARGE_CLUSTER/str(cur_cid))
    
    logger.info(f'Sucessfully convert and dump clusters to: {DP_MARGE_CLUSTER}')


if __name__ == "__main__":
    convert()
