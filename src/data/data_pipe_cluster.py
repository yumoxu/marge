# -*- coding: utf-8 -*-
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join, isdir, dirname, abspath
import sys

import utils.config_loader as config
from utils.config_loader import path_parser, logger, config_model, config_meta
from data.dataset_parser import dataset_parser
import data.data_tools as data_tools

import utils.tools as tools

sys.path.insert(0, dirname(dirname(abspath(__file__))))


class ClusterDataset(Dataset):
    def __init__(self, cid, query, transform=None):
        super(ClusterDataset, self).__init__()
        self.doc_ids = tools.get_doc_ids(cid, remove_illegal=True)  # remove empty docs with no preprocessed sents
        self.query = query
        self.yy = 0.0  # 0.0

        self.transform = transform
        self.bert_in_func = data_tools.get_bert_in_func()

    def __len__(self):
        return len(self.doc_ids)

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
        year, cc, fn = self.doc_ids[index].split(config.SEP)

        doc_fp = join(path_parser.data_docs, year, cc, fn)

        # build xx
        xx = self.bert_in_func(self.query, doc_fp=doc_fp)

        # build yy
        yy = self._vec_label(self.yy)

        sample = {
            **xx,
            'yy': yy,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ClusterDataLoader(DataLoader):
    def __init__(self, cid, query, transform=data_tools.ToTensor()):
        dataset = ClusterDataset(cid, query)
        self.transform = transform
        self.cid = cid

        super(ClusterDataLoader, self).__init__(dataset=dataset,
                                                # batch_size=config_model['d_batch'],
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=3,  # 3
                                                drop_last=False)

    def _generator(self, super_iter):
        while True:
            # try:
            batch = next(super_iter)
            # for func in self.transform:
            batch = self.transform(batch)
            yield batch

    def __iter__(self):
        super_iter = super(ClusterDataLoader, self).__iter__()
        return self._generator(super_iter)


class QSDataLoaderOneClusterABatch:
    """
        iter over all clusters.
        each cluster is handled with a separate data loader.
    """

    def __init__(self):
        pos_narr_dict = dataset_parser.build_query_info(config_meta['test_year'], tokenize=None)
        # pos_headline_dict = dataset_parser.build_headline_info(config_meta['test_year'], tokenize=None, silent=True)

        self.loader_init_params = []

        for cid in pos_narr_dict:
            narr = pos_narr_dict[cid][config.NARR]
            self.loader_init_params.append({
                'cid': cid,
                'narr': narr,
            })

    def _loader_generator(self):
        for params in self.loader_init_params:
            c_loader = ClusterDataLoader(**params)
            for batch_idx, batch in enumerate(c_loader):  # should be only one batch / $n_docs$ batches in a loader
                # logger.info('batch_idx: {}'.format(batch_idx))
                yield {
                    'cid': params['cid'],
                    'batch': batch,
                }

    def __iter__(self):
        return self._loader_generator()


class QSDataLoader:
    """
        iter over all clusters.
        each cluster is handled with a separate data loader.

        tokenize_narr: whether tokenize query into sentences.
    """

    def __init__(self, tokenize_narr, query_type=None):
        if query_type == config.TITLE:
            query_dict = dataset_parser.get_cid2title()
        elif query_type == config.NARR:
            query_dict = dataset_parser.get_cid2narr()
        else:
            query_dict = dataset_parser.get_cid2query(tokenize_narr)

        cids = tools.get_test_cc_ids()

        self.loader_init_params = []
        for cid in cids:
            query = query_dict[cid]
            self.loader_init_params.append({
                'cid': cid,
                'query': query,
            })

    def _loader_generator(self):
        for params in self.loader_init_params:
            c_loader = ClusterDataLoader(**params)
            yield c_loader

    def __iter__(self):
        return self._loader_generator()


class GSQSDataLoader:
    """
        for general-specific data use.

        iter over all clusters.
        each cluster is handled with a separate data loader.
    """
    def __init__(self):
        cid2trigger = dataset_parser.get_cid2trigger(tokenize_narr=True)
        cids = tools.get_test_cc_ids()

        self.loader_init_params = []
        for cid in cids:
            query = cid2trigger[cid]['query']
            self.loader_init_params.append({
                'cid': cid,
                'query': query,
                # 'join_query_para': join_query_para,
            })

    def _loader_generator(self):
        for params in self.loader_init_params:
            c_loader = ClusterDataLoader(**params)
            yield c_loader

    def __iter__(self):
        return self._loader_generator()
