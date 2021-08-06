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

from tools.query_tools import get_cid2masked_query
import utils.config_loader as config
from utils.config_loader import config_model, path_parser, logger
from utils.tools import get_test_cc_ids, get_query_w_cid
from data.dataset_parser import dataset_parser
import data.data_tools as data_tools
from bert_rr.bert_input import build_bert_sentence_x

import io
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
    This module contains classes for dataset and data loader.

    STATUS: *To TEST*
    
"""

def load_retrieved_sentences(retrieved_dp, cid):
    """
        This func was mainly copied from ir.ir_tools but has been changed as follows:

            Origin: 
                processed_sents = [dataset_parser._proc_sent(ss, rm_dialog=True, rm_stop=True, stem=True)
                       for ss in original_sents]
            Now:
                processed_sents = [dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)
                       for ss in original_sents]
            
            Reason of change: you can filter DIALOG sentences during IR. 

    :param retrieved_dp:
    :param cid:
    :return:
    """
    if not exists(retrieved_dp):
        raise ValueError('retrieved_dp does not exist: {}'.format(retrieved_dp))

    fp = join(retrieved_dp, cid)
    with io.open(fp, encoding='utf-8') as f:
        content = f.readlines()

    original_sents = [ll.rstrip('\n').split('\t')[-1] for ll in content]

    processed_sents = [dataset_parser._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)
                       for ss in original_sents]

    return [original_sents], [processed_sents]  # for compatibility of document organization for similarity calculation


def load_retrieved_sentences_and_sids(retrieved_dp, cid):
    """
        Load with sids for rank with local/global pos.

    :param retrieved_dp:
    :param cid:
    :return:
    """
    if not exists(retrieved_dp):
        raise ValueError(f'retrieved_dp does not exist: {retrieved_dp}')

    fp = join(retrieved_dp, cid)
    with io.open(fp, encoding='utf-8') as f:
        content = f.readlines()
    
    original_sents = []
    processed_sents = []
    for ll in content:
        sid, _, sentence = ll.rstrip('\n').split('\t')
        original_sents.append((sid, sentence))

        proc_sentence = dataset_parser._proc_sent(sentence, rm_dialog=False, rm_stop=True, stem=True)
        processed_sents.append((sid, proc_sentence))

    return [original_sents], [processed_sents]  # for compatibility of document organization for similarity calculation


class ClusterDataset(Dataset):
    def __init__(self, cid, query, retrieve_dp, with_sub, transform=None, silent=False):
        super(ClusterDataset, self).__init__()
        original_sents, _ = load_retrieved_sentences(retrieved_dp=retrieve_dp, cid=cid)
        self.sentences = original_sents[0]

        self.query = query
        self.yy = 0.0  # 0.0

        self.with_sub = with_sub
        self.transform = transform
        self.silent = silent

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def _vec_label(yy):
        if yy == '-1.0':
            yy = 0.0
        return np.array([yy], dtype=np.float32)

    def __getitem__(self, index):
        """
            get an item from self.doc_ids.

            return a sample: (xx, yy)
        """
        # build xx
        xx, tokens = build_bert_sentence_x(self.query, sentence=self.sentences[index], with_sub=self.with_sub)

        # build yy
        yy = self._vec_label(self.yy)

        sample = {
            **xx,
            'yy': yy,
        }

        if self.transform:
            sample = self.transform(sample)
        
        if index < 5 and not self.silent:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (index))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("token_ids: %s" % " ".join([str(x) for x in xx['token_ids']]))

        return sample


class ClusterDataLoader(DataLoader):
    def __init__(self, cid, query, retrieve_dp, with_sub, transform=data_tools.ToTensor(), silent=False):
        dataset = ClusterDataset(cid, query, retrieve_dp=retrieve_dp, with_sub=with_sub, silent=silent)
        self.transform = transform
        self.cid = cid

        super(ClusterDataLoader, self).__init__(dataset=dataset,
                                                batch_size=64,
                                                shuffle=False,
                                                num_workers=5,  # 3
                                                drop_last=False)

    def _generator(self, super_iter):
        while True:
            batch = next(super_iter)
            batch = self.transform(batch)
            yield batch

    def __iter__(self):
        super_iter = super(ClusterDataLoader, self).__iter__()
        return self._generator(super_iter)


class QSDataLoader:
    """
        iter over all clusters.
        each cluster is handled with a separate data loader.

        tokenize_narr: whether tokenize query into sentences.
    """

    def __archived_init__(self, tokenize_narr, query_type, retrieve_dp, with_sub):
        if query_type:
            query_dict = get_cid2masked_query(query_type, with_sub=with_sub)

        cids = get_test_cc_ids()

        self.loader_init_params = []
        for cid in cids:
            if query_type:
                query = get_query_w_cid(query_dict, cid=cid)
            else:
                query = None
            # query = query_dict[cid]
            self.loader_init_params.append({
                'cid': cid,
                'query': query,
                'retrieve_dp': retrieve_dp,
                'with_sub': with_sub,
            })

    def __init__(self, tokenize_narr, query_type, retrieve_dp, with_sub):
        cids = get_test_cc_ids()
        self.loader_init_params = []
        self.query_type = query_type

        if self.query_type:
            if '@' in self.query_type:
                self.query_dict = get_cid2masked_query(config.QUERY, with_sub=with_sub)  # use the whole query
                _, mask_dataset, mask_fn = self.query_type.split('@')
                masked_query_fp = path_parser.pred / mask_dataset / f'{mask_fn}.json'
                self.cid2masked_query = self.load_cid2masked_query(masked_query_fp)
            else:
                self.query_dict = get_cid2masked_query(self.query_type, with_sub=with_sub)

        for cid in cids:
            query = self.get_query(cid)
            self.loader_init_params.append({
                'cid': cid,
                'query': query,
                'retrieve_dp': retrieve_dp,
                'with_sub': with_sub,
            })
    
    def load_cid2masked_query(self, masked_query_fp):
        cid2masked_query = {}
        for line in open(masked_query_fp).readlines():
            json_obj = json.loads(line.strip('\n')) 
            cid2masked_query[json_obj['cid']] = ' '.join(json_obj['masked_seq'])
        return cid2masked_query

    def get_query(self, cid):
        if not self.query_type:
            return None
        
        query = get_query_w_cid(self.query_dict, cid=cid)
        if '@' not in self.query_type:
            return query

        query += self.cid2masked_query[cid]  # append
        return query

    def _loader_generator(self):
        for params in self.loader_init_params:
            c_loader = ClusterDataLoader(**params)
            yield c_loader

    def __iter__(self):
        return self._loader_generator()


class Eli5TdqfsQSDataLoader:
    """
        iter over all clusters.
        each cluster is handled with a separate data loader.
    """

    def __init__(self, test_cid_query_dicts, query_type, retrieve_dp, with_sub, qe=False, cid2qe=None):
        """
            cid2qe: for iterative rank and sel.
                Queries are expanded every iteration with the top result from the last iteration.
        """
        self.loader_init_params = []
        self.query_type = query_type
        
        if self.query_type and '@' in self.query_type:
            _, mask_dataset, mask_fn = self.query_type.split('@')
            masked_query_fp = path_parser.pred / mask_dataset / f'{mask_fn}.json'
            self.cid2masked_query = self.load_cid2masked_query(masked_query_fp)

        for idx, cq_dict in enumerate(test_cid_query_dicts):
            silent = False if idx == 0 else True
            cid, query = cq_dict['cid'], cq_dict['query']
            query = self.proc_query(query=query, cid=cid)
            if qe and cid2qe:
                expansion = cid2qe[cid]
                if not expansion:
                    break
                query += ' ' + expansion

            self.loader_init_params.append({
                'cid': cid,
                'query': query,
                'retrieve_dp': retrieve_dp,
                'with_sub': with_sub,
                'silent': silent,
            })

    def load_cid2masked_query(self, masked_query_fp):
        cid2masked_query = {}
        for line in open(masked_query_fp).readlines():
            json_obj = json.loads(line.strip('\n')) 
            cid2masked_query[json_obj['cid']] = ' '.join(json_obj['masked_seq'])
        return cid2masked_query

    def proc_query(self, query, cid=None):
        if not self.query_type:
            return None

        if self.query_type == config.QUERY:
            return query
        
        if self.query_type == 'add_mask':
            return '[MASK] ' + query + ' [MASK]'

        if self.query_type == 'add_left_mask':
            return '[MASK] ' + query

        if self.query_type == 'add_left_mask_right_dot':
            return '[MASK] ' + query + ' .'

        if '@' in self.query_type:
            query =  '[MASK] ' + query + ' . ' + self.cid2masked_query[cid]
            return query

        raise ValueError(f'Query type not supported: {self.query_type}')

    def _loader_generator(self):
        for params in self.loader_init_params:
            c_loader = ClusterDataLoader(**params)
            yield c_loader

    def __iter__(self):
        return self._loader_generator()
