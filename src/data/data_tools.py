# -*- coding: utf-8 -*-
import torch
from os.path import join, isdir, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import utils.config_loader as config
from utils.config_loader import path_parser, logger, config_model, config_meta

def get_bert_in_func():
    if config.meta_model_name == 'bert_qa':
        from bert_qa import bert_input
        bert_in_func = bert_input.build_bert_x

    elif config.meta_model_name == 'bert_mb':
        from bert_mb import bert_input
        bert_in_func = bert_input.build_bert_x

    elif config.meta_model_name == 'bert_mil':
        from bert_mil import bert_input
        bert_in_func = bert_input.build_bert_x

    elif config.join_query_para:
        from data import bert_input
        bert_in_func = bert_input.build_bert_x
    else:
        from data import bert_input_sep
        bert_in_func = bert_input_sep.build_bert_x_sep

    logger.info('Using bert_in_func: {}'.format(config.meta_model_name))
    return bert_in_func


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
    """

    def __call__(self, numpy_dict):
        for (k, v) in numpy_dict.items():
            # logger.info('[BEFORE TO TENSOR] type of {0}: {1}'.format(k, v.dtype))

            if k.endswith('_ids'):
                v = v.type(torch.LongTensor)  # for embedding look up
                # logger.info('[TO LONG TENSOR] convert {0} => {1}'.format(k, v.dtype))

            if 'placement' in config_meta and config_meta['placement'] != 'cpu':
                # origin_type = v.type()
                v = v.cuda()
                # logger.info('[TO CUDA TENSOR] {0}: {1} => {2}'.format(k, origin_type, v.type()))

            numpy_dict[k] = v

        # logger.info('is cuda available: {}'.format(torch.cuda.is_available()))
        # for (k, v) in numpy_dict.items():
        #     # logger.info('[BEFORE TO TENSOR] type of {0}: {1}'.format(k, v.dtype))
        #     if type(v) == np.ndarray:
        #         v = torch.from_numpy(v)
        #
        #         if k.endswith('_ids'):
        #             v = v.type(torch.LongTensor)  # for embedding look up
        #             logger.info('[TO TENSOR] convert {0} => {1}'.format(k, v.dtype))
        #
        #         if config.placement in ('auto', 'single'):
        #             v = v.cuda()
        #             logger.info('[TO CUDA] type of {0}: {1}'.format(k, v.dtype))
        #
        #     numpy_dict[k] = v

        return numpy_dict


def get_data_loader(model_mode):
    """
        Get data loader as per model name.
    """
    if config.meta_model_name == 'bert_mil':
        from bert_mil.data_pipe import QueryNetDataLoader
        return QueryNetDataLoader(model_mode=model_mode)
    else:
        from data.data_pipe import QueryNetDataLoader
        return QueryNetDataLoader(model_mode=model_mode)
