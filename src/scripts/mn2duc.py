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

N_SENTS = 1  # 1, 4, 8
MIN_NW_MODE = 'sample'  # max
METRIC = 'rouge_2_f1'

FP_JSON_TEST = path_parser.data / 'top_mn' / f'top_mn_{MIN_NW_MODE}_{N_SENTS}_f1' / 'test.json'

"""
    This module turn MN data into DUC format, i.e., organize sentences and summaries via Cluster. 
"""

def mn2duc(proc_newline):
    DP_TEST = path_parser.data / 'test_mn_cluster'

    lines = open(FP_JSON_TEST).readlines()
    for line in tqdm(lines):
        json_obj = json.loads(line)
        cid = str(json_obj['cid'])
        passage_objs = json_obj['passages']

        if proc_newline:
            passages_items = [[po['id'], str(po[METRIC]), po['passage'].replace('NEWLINE_CHAR', '').strip()] for po in passage_objs]
        else:
            passages_items = [[po['id'], str(po[METRIC]), po['passage']] for po in passage_objs]
        passage_records = ['\t'.join(tuple) for tuple in passages_items]
        content = '\n'.join(passage_records)
        io.open(DP_TEST/cid, 'a').write(content)


def mn_summary2duc():
    SUMMARY_DP = path_parser.data / 'test_mn_summary'
    mn_parser = MultiNewsParser(dataset_var='test')

    sample_generator = mn_parser.sample_generator()
    for cluster_idx, _, summary in tqdm(sample_generator, total=5622):
        with open(SUMMARY_DP / str(cluster_idx), 'a') as f:
            f.write(summary)


if __name__ == "__main__":
    mn2duc(proc_newline=True)
    # mn_summary2duc()

