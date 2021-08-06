# -*- coding: utf-8 -*-
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join, isdir, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import utils.config_loader as config
from utils.config_loader import path_parser, logger, config_model, config_meta
if config_meta['general_specific']:
    from data.dataset_index_builder_gs import data_indexer
else:
    from data.dataset_index_builder import data_indexer

from data.dataset_parser import dataset_parser
import data.bert_input as bert_in
import data.bert_input_sep as bert_input_sep
import data.bert_input_sl as bert_input_sl
import data.data_tools as data_tools
import utils.tools as tools


class To2DMat(object):
    def __call__(self, numpy_dict):
        for (k, v) in numpy_dict.items():
            # logger.info('[BEFORE TO TENSOR] type of {0}: {1}'.format(k, v.dtype))
            if k in ('token_ids', 'seg_ids', 'token_masks'):
                numpy_dict[k] = v.reshape(-1, config_model['max_n_tokens'])

        return numpy_dict


class QVDataset(Dataset):
    def __init__(self, transform=None):
        super(QVDataset, self).__init__()
        self.index_dicts = data_indexer.load_end_to_end(config_meta['test_year'], config_meta['mode'])
        self.transform = transform

    def __len__(self):
        return len(self.index_dicts)

    @staticmethod
    def _vec_label(yy):
        return np.array([yy], dtype=np.float32)

    def __getitem__(self, index):
        """
            get an item from self.index_dicts.
            keys: 'doc_id', 'query_type', 'neg_label', 'y', 'query'
        """
        index_dict = self.index_dicts[index]
        year, cc, fn = index_dict['doc_id'].split(config.SEP)

        doc_fp = join(path_parser.data_docs, year, cc, fn)

        yy = self._vec_label(index_dict['y'])

        query_res = dataset_parser.parse_content(content=index_dict['query'], clip_func=None)
        doc_res = dataset_parser.parse_article(fp=doc_fp, concat_paras=True)

        sample = {'yy': yy,
                  'query_wids': query_res['word_ids'],
                  'query_sent_mask': query_res['sent_mask'],
                  'query_word_mask': query_res['word_mask'],
                  'doc_wids': doc_res['word_ids'],
                  'doc_sent_mask': doc_res['sent_mask'],
                  'doc_word_mask': doc_res['word_mask'],
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class BERTQVDataset(Dataset):
    def __init__(self, model_mode, transform=None):
        super(BERTQVDataset, self).__init__()
        self.index_dicts = data_indexer.load(model_mode)
        self.transform = transform
        self.bert_in_func = data_tools.get_bert_in_func()

    def __len__(self):
        return len(self.index_dicts)

    @staticmethod
    def _vec_label(yy):
        if config.label_mode == 'continuous':
            if config_model['score_func'] == 'tanh':
                raise ValueError('Continuous label mode should only be used with sigmoid score func.'
                                 'Set score func to sigmoid OR set label mode to binary!')
            return np.array([yy], dtype=np.float32)

        elif config.label_mode == 'binary':
            if config_model['score_func'] == 'sigmoid' and yy == '-1.0':
                yy = 0.0

            return np.array([yy], dtype=np.float32)

        else:
            raise ValueError('Invalid label_mode: {}'.format(config_meta['label_mode']))

    def _reverse_yy(self, yy):
        assert config.label_mode == 'binary'

        return 1.0 - yy

    def __getitem__(self, index):
        """
            get an item from self.index_dicts.
            keys: 'doc_id', 'query_type', 'neg_label', 'y', 'query'

            return a sample: (xx, yy)
                xx:
                    'token_ids', 'seg_ids',
                    'token_masks',
                    'query_sent_masks', 'para_sent_masks',
                    'para_masks',
                    'doc_masks'
        """
        index_dict = self.index_dicts[index]
        year, cc, fn = index_dict['doc_id'].split(config.SEP)

        doc_fp = join(path_parser.data_docs, year, cc, fn)

        # build xx
        xx = self.bert_in_func(index_dict['query'], doc_fp)

        # build yy
        yy = self._vec_label(index_dict['y'])

        # query: 0.0, headline: 1.0
        yy = self._reverse_yy(yy)

        sample = {
            **xx,
            'yy': yy,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class BERTQVDatasetSL(Dataset):
    def __init__(self, model_mode, transform=None):
        super(BERTQVDatasetSL, self).__init__()
        self.index_dicts = data_indexer.load(model_mode)
        self.transform = transform

    def __len__(self):
        return len(self.index_dicts)

    @staticmethod
    def _vec_label(yy):
        if config.label_mode == 'continuous':
            if config_model['score_func'] == 'tanh':
                raise ValueError('Continuous label mode should only be used with sigmoid score func.'
                                 'Set score func to sigmoid OR set label mode to binary!')
            return np.array([yy], dtype=np.float32)

        elif config.label_mode == 'binary':
            if config_model['score_func'] == 'sigmoid' and yy == '-1.0':
                yy = 0.0

            return np.array([yy], dtype=np.float32)

        else:
            raise ValueError('Invalid label_mode: {}'.format(config_meta['label_mode']))

    def _reverse_yy(self, yy):
        assert config.label_mode == 'binary'

        return 1.0 - yy

    def __getitem__(self, index):
        """
            get an item from self.index_dicts.
            keys: 'doc_id', 'query_type', 'neg_label', 'y', 'query'

            return a sample: (xx, yy)
                xx:
                    'token_ids', 'seg_ids',
                    'token_masks',
                    'query_sent_masks', 'para_sent_masks',
                    'para_masks',
                    'doc_masks'
        """
        index_dict = self.index_dicts[index]
        year, cc, fn = index_dict['doc_id'].split(config.SEP)

        doc_fp = join(path_parser.data_docs, year, cc, fn)

        # build xx
        xx_doc = bert_input_sl.build_bert_x_doc_sl(doc_fp)
        xx_trigger = bert_input_sl.build_bert_x_trigger_sl(trigger=index_dict['query'])

        # build yy
        yy = self._vec_label(index_dict['y'])

        # query: 0.0, headline: 1.0
        yy = self._reverse_yy(yy)

        sample = {
            **xx_doc,
            **xx_trigger,
            'yy': yy,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class QueryNetDataLoader(DataLoader):
    def __init__(self, model_mode, transform=data_tools.ToTensor()):
        """

        :param model_mode: [train | dev | test | debug]
        :return:
        """
        if config.data_mode == 'mixed':
            logger.info('[DATA LOADER]: mixed {0} data'.format(model_mode))
        elif config.data_mode == 'sep':
            logger.info('[DATA LOADER]: {0} data for {1}'.format(model_mode, config_meta['test_year']))
        else:
            raise ValueError('Invalid data_mode: {}'.format(config.data_mode))

        # if config.meta_model_name.endswith('sl'):
        #     dataset = BERTQVDatasetSL(model_mode)
        # else:
        dataset = BERTQVDataset(model_mode)

        self.transform = transform

        drop_last = False
        shuffle = False

        if model_mode == 'train':
            drop_last = True
            shuffle = True

        super(QueryNetDataLoader, self).__init__(dataset=dataset,
                                                 batch_size=config_model['d_batch'],
                                                 shuffle=shuffle,
                                                 num_workers=3,  # 3
                                                 drop_last=drop_last)

    def _generator(self, super_iter):
        while True:
            batch = next(super_iter)
            batch = self.transform(batch)
            yield batch

    def __iter__(self):
        super_iter = super(QueryNetDataLoader, self).__iter__()
        return self._generator(super_iter)



if __name__ == '__main__':
    data_loader = QueryNetDataLoader(model_mode='train')

    # data_loader = QSDataLoader()
    for batch_idx, batch in enumerate(data_loader):
        logger.info('batch_idx: {}'.format(batch_idx))
        logger.info('Size of y: {}'.format(batch['yy'].size()))
        # logger.info('y: {}'.format(batch['yy']))
        if config.join_query_para:
            logger.info('Size of token_ids: {}'.format(batch['token_ids'].size()))
            logger.info('Size of seg_ids: {}'.format(batch['seg_ids'].size()))
        else:
            logger.info('Size of para_token_ids: {}'.format(batch['para_token_ids'].size()))
            logger.info('Size of para_seg_ids: {}'.format(batch['para_seg_ids'].size()))

            logger.info('Size of query_token_ids: {}'.format(batch['query_token_ids'].size()))
            logger.info('Size of query_seg_ids: {}'.format(batch['query_seg_ids'].size()))

    # data_loader_generator = QSDataLoader()
    # for data_loader in data_loader_generator:
    #     cid = data_loader.cid
    #     for batch_idx, batch in enumerate(data_loader):
    #         logger.info('batch_idx: {0} for {1}'.format(batch_idx, cid))
    #         logger.info('Size of y: {}'.format(batch['yy'].size()))
    #         logger.info('Size of token_ids: {}'.format(batch['token_ids'].size()))
    #         logger.info('Size of seg_ids: {}'.format(batch['seg_ids'].size()))
