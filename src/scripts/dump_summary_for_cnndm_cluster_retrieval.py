# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
import logging

import json
from tqdm import tqdm
import shutil
from multiprocessing import Pool
from pyrouge import Rouge155

import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import utils.tools as tools
from data.cnndm_parser import CnnDmParser


if config.path_type == 'afs':
    DP_PROJ = Path('/disk/nfs/ostrom/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_DATA = DP_PROJ / 'data'
DP_SUMMARY_CNNDM = DP_DATA / 'cnndm' / 'raw_cnndm_summary'

DATASET_VARS = ['train', 'val', 'test', 'debug']

CNNDM_DATASET_SIZE = {
    'train': 287227,
    'val': 13368,
    'test': 11490,
}

"""
    Build summaries from CNN/DM for the index and retrieval DrQA.

"""

def dump_summary_cnndm(dataset_var):
    """
        Build a json file containing all CNN/DM summaries.

        This serves as the input to DrQA, based on which a sqlite DB is built for retrieval.

    """
    if not exists(DP_SUMMARY_CNNDM):
        os.mkdir(DP_SUMMARY_CNNDM)

    if dataset_var not in DATASET_VARS:
        raise ValueError(f'Illegal {dataset_var}!')

    total = CNNDM_DATASET_SIZE[dataset_var]
    dump_fp = DP_SUMMARY_CNNDM / f'{dataset_var}.json'

    if exists(dump_fp):
        raise ValueError(f'Remove {dump_fp} before dumping data')
    
    logger.info(f'Ready to dump json data to {dump_fp}')

    parser = CnnDmParser(dataset_var=dataset_var)
    sample_generator = parser.sample_generator()

    with open(dump_fp, 'a') as f:
        for doc_idx, doc_sentences, summary in tqdm(sample_generator, total=total):
            data = {
                'id': str(doc_idx),
                'text': summary,
            }
            json_str = json.dumps(data, ensure_ascii=False)
            f.write(json_str+'\n')


if __name__ == "__main__":
    dataset_var = 'train'
    dump_summary_cnndm(dataset_var=dataset_var)
