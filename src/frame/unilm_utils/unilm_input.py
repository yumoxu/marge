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

from pathlib import Path
import copy
import numpy as np
import json
import io
from os import listdir
from tqdm import tqdm

import utils.config_loader as config
from utils.config_loader import path_parser
import utils.tools as tools
import querysum.ir.ir_tools as ir_tools
import tools.query_tools as query_tools
# import querysum.bert_marge.marge_config as marge_config


class UniLMInput:
    def __init__(self, marge_config, rank_dp=None, text_dp=None, 
            fix_input=True, cluster_ids=None, 
            prepend_len=False, prepend_query=None,
            test_cid_query_dicts=None):
        """
            Build UniLM input from rand_dp or text_dp. 

            marge_config should define:
                - POSITIONAL
                - UNILM_IN_FILE_NAME
                - MARGE_MODEL_NAME_BERT
                - FILTER_VAR
                - FILTER

            fix_input: fix the mess in the input sentences, if set True.
        """
        super().__init__()
        self.marge_config = marge_config
        self.rank_dp =rank_dp
        self.text_dp =text_dp
        self.fix_input = fix_input
        self.tgt = 'TARGET_PLACEHOLDER'
        self.multi_pass = False

        self.cluster_ids = cluster_ids
        self.prepend_len = prepend_len
        self.prepend_query = prepend_query
        
        if prepend_query:
            if not test_cid_query_dicts:
                if prepend_query == 'raw':
                    self.query_dict = query_tools.get_cid2raw_query(query_type=config.QUERY)
                elif prepend_query == 'masked':
                    self.query_dict = query_tools.get_cid2masked_query(query_type=config.QUERY, with_sub=False)
                else:
                    raise ValueError(f'Invalid prepend_query: {prepend_query}')
            else:
                self.query_dict = {}  # for TD-QFS

                if '@' in self.prepend_query:
                    _, mask_dataset, mask_fn = self.prepend_query.split('@')
                    masked_query_fp = path_parser.pred / mask_dataset / f'{mask_fn}.json'
                    self.cid2masked_query = self.load_cid2masked_query(masked_query_fp)

                for cq_dict in test_cid_query_dicts:
                    query = self.proc_query(query=cq_dict['query'], cid=cq_dict['cid'])
                    self.query_dict[cq_dict['cid']] = query

    def load_cid2masked_query(self, masked_query_fp):
        cid2masked_query = {}
        for line in open(masked_query_fp).readlines():
            json_obj = json.loads(line.strip('\n')) 
            cid2masked_query[json_obj['cid']] = ' '.join(json_obj['masked_seq'])
        return cid2masked_query

    def proc_query(self, query, cid):
        """
            For TDQFS.
        """
        if self.prepend_query == config.QUERY:
            return query
        
        if self.prepend_query == 'add_mask':
            return '[MASK] ' + query + ' [MASK]'

        if self.prepend_query == 'add_left_mask':
            return '[MASK] ' + query

        if self.prepend_query == 'add_left_mask_right_dot':
            return '[MASK] ' + query + ' .'
        
        if '@' in self.prepend_query:
            query =  '[MASK] ' + query + ' . ' + self.cid2masked_query[cid]
            return query

        raise ValueError(f'Query type not supported: {self.prepend_query}')


    def order(self, items):
        if self.marge_config.POSITIONAL == 'global': 
            items = [(rank, item[0].split('_')[0], int(item[0].split('_')[1]), item[1], item[2]) for rank, item in enumerate(items)]
            ordered_items = sorted(items, key=lambda item: item[2], reverse=False)  # item[2] is sentence index
            return ordered_items
        
        elif self.marge_config.POSITIONAL == 'local':  # TODO to be tested
            items = [(rank, item[0].split('_')[0], int(item[0].split('_')[1]), item[1], item[2]) for rank, item in enumerate(items)]

            # index
            did2items = {}
            for item in items:
                did = item[1]
                if did in did2items:
                    did2items[did].append(item)
                else:
                    did2items[did] = [item]

            # order docs as per relevance
            did2score = {}
            for did, items in did2items.items():
                did2score[did] = sum([float(item[-2]) for item in items]) / len(items)
            sorted_did2score = sorted(did2score.items(), key=lambda item: item[1], reverse=True)  #  descent
            sorted_doc_ids = [did for did, score in sorted_did2score]

            # order sentences inside doc as per postion
            ordered_items = []
            for did in sorted_doc_ids:
                doc_items = sorted(did2items[did], key=lambda item: item[2], reverse=False)  # item[2] is sentence index; ascent
                ordered_items.extend(doc_items)
            return ordered_items
        
        else:
            raise NotImplementedError('local mode has not been implemented')

    def fix_customized(self, sent):
        while True:
            i = 0
            while i < len(sent):
                if i != '-':
                    continue

                end = i + 1
                while end < len(sent):
                    if end == '-':
                        end += 1
                    else:
                        break
                
                if end - start > 2:
                    sent[start_idx:end_idx] = '-'
                    break
                else:
                    continue
                i = end
            
            if i == len(sent) -1:
                break
                
    def fix(self, sent):
        sent = sent.strip().strip('-')
        sent = sent.replace('---', '')
        return sent

    def get_generation_unit_ids(self):
        """
            A generation unit is:
                - a cluster, when self.multi_pass == False, or
                - a slot, when self.multi_pass == True.

        """
        # cluster_ids = tools.get_test_cc_ids()

        if not self.multi_pass:
            return self.cluster_ids

        slot_ids = []
        if self.rank_dp:
            fns = [fn for fn in listdir(self.rank_dp) if isfile(join(self.rank_dp, fn))]
        else:
            fns = [fn for fn in listdir(self.text_dp) if isfile(join(self.text_dp, fn))]
        
        for cid in self.cluster_ids:
            # order slot_ids
            # max_slot_idx = max([int(fn.split('_')[-1]) for fn in fns if fn.startswith(cid)])
            # _slot_ids = [f'{cid}_{si}'  for si in range(max_slot_idx+1)]
            _slot_indices = [int(fn.split('_')[-1]) for fn in fns if fn.startswith(cid)]
            _slot_indices.sort()
            _slot_ids = [f'{cid}_{si}' for si in _slot_indices]
            
            slot_ids.extend(_slot_ids)
        
        return slot_ids
        
    def build_from_rank(self):
        """
            For:
                - each cluster, when self.multi_pass == False, or
                - each slot in all clusters, when self.multi_pass == True, 
            concatenate the topK sentences selected by MaRGE to build UniLM input

        """
        unit_ids = self.get_generation_unit_ids()
        out_fp = path_parser.unilm_in / self.marge_config.UNILM_IN_FILE_NAME

        json_str = ''
        
        if hasattr(self.marge_config, 'MARGE_MODEL_NAME_BERT'):
            model_name = self.marge_config.MARGE_MODEL_NAME_BERT
        elif hasattr(self.marge_config, 'RR_MODEL_NAME_BERT'):
            model_name = self.marge_config.RR_MODEL_NAME_BERT
        elif hasattr(self.marge_config, 'RR_SUBQ_MODEL_NAME_BERT'):
            model_name = self.marge_config.RR_SUBQ_MODEL_NAME_BERT
        elif hasattr(self.marge_config, 'QA_MODEL_NAME_BERT'):
            model_name = self.marge_config.QA_MODEL_NAME_BERT
        elif hasattr(self.marge_config, 'IR_MODEL_NAME_TF'):
            model_name = self.marge_config.IR_MODEL_NAME_TF
        else:
            raise ValueError('Invalid config file')

        for uid in tqdm(unit_ids):
            retrieval_params = {
                'model_name': model_name,
                'cid': uid,
                'filter_var': self.marge_config.FILTER_VAR,
                'filter': self.marge_config.FILTER,
                'deduplicate': None,
                'min_ns': None,  # self.marge_config.MARGE_MIN_NS
                'rank_dp': self.rank_dp,
            }

            retrieved_items = ir_tools.retrieve(**retrieval_params)
            
            if self.marge_config.POSITIONAL:
                retrieved_items = self.order(retrieved_items)
                # lines = ['\t'.join([str(data) for data in item]) for item in retrieved_items]
                # text = '\n'.join(lines)
                # print(text)
                # return None
            
            sentences = [item[-1] for item in retrieved_items]
            if self.fix_input:
                sentences = [self.fix(sent) for sent in sentences]

            # text = '\n'.join(sentences)
            # print(text)
            # return None
            src = ' '.join(sentences)
            
            if self.prepend_query and self.multi_pass:
                raise ValueError('Cannot prepend_query for subqueries now. Have not implemented.')

            if self.prepend_query:
                query = self.query_dict[uid]
                src = query + ' [SEP] ' + src
                
            if self.prepend_len:
                src = '[unused250] ' + src

            json_obj = {
                "uid": uid,
                "src": src,
                "tgt": self.tgt,
            }
            _j_str = json.dumps(json_obj, ensure_ascii=False)
            json_str += f'{_j_str}\n'
        
        with open(out_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(json_str)
        print(f'Finshed UniLM input file: {out_fp}')

    def build_from_text(self):
        """
            For:
                - each cluster, when self.multi_pass == False, or
                - each slot in all clusters, when self.multi_pass == True, 
            concatenate the topK sentences selected by MaRGE to build UniLM input

        """
        unit_ids = self.get_generation_unit_ids()
        out_fp = path_parser.unilm_in / self.marge_config.UNILM_IN_FILE_NAME

        json_str = ''
        for uid in tqdm(unit_ids):
            fp = Path(self.text_dp) / uid
            print(f'load {fp}')
            lines = io.open(fp).readlines()
            sentences = [ll.strip('\n') for ll in lines]
            print(f'\tLine: {sentences[0]}')
            
            if self.fix_input:
                sentences = [self.fix(sent) for sent in sentences]
            
            src = ' '.join(sentences)
            
            if self.prepend_len:
                src = '[unused250] ' + src

            json_obj = {
                "uid": uid,
                "src": src,
                "tgt": self.tgt,
            }
            _j_str = json.dumps(json_obj, ensure_ascii=False)
            json_str += f'{_j_str}\n'
        
        with open(out_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(json_str)


