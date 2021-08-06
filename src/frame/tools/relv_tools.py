import os
from os.path import exists, join, dirname, abspath
import sys
from tqdm import tqdm

import utils.tools as tools
import utils.graph_io as graph_io
import utils.config_loader as config
from utils.config_loader import config_summ, logger, path_parser
import summ.rank_sent as rank_sent
import summ.select_sent as select_sent

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def rank_end2end(model_name, n_iter=None):
    rank_dp_params = {
        'model_name': model_name,
        'n_iter': None,
        'extra': None,
        **config_summ,
    }
    rank_dp = tools.get_rank_dp(**rank_dp_params)
    if exists(rank_dp):
        raise ValueError('rank_dp exists: {}'.format(rank_dp))
    os.mkdir(rank_dp)

    dp_mode = 'r'
    dp_params = {
        'model_name': config.model_name,  # one model has only one suit of summary components but different ranking sys
        'n_iter': n_iter,
        **config_summ,
        'mode': dp_mode,
    }

    summ_comp_root = graph_io.get_summ_comp_root(**dp_params)
    rel_vec_dp = graph_io.get_rel_vec_dp(summ_comp_root, mode=dp_mode)
    sid2abs_dp = graph_io.get_sid2abs_dp(summ_comp_root, mode=dp_mode)

    cc_ids = tools.get_test_cc_ids()

    for cid in tqdm(cc_ids):
        rel_vec = graph_io.load_rel_vec(rel_vec_dp, cid)
        sid2abs = graph_io.load_sid2abs(sid2abs_dp, cid)

        abs2sid = {}
        for sid, abs in sid2abs.items():
            abs2sid[abs] = sid

        sid2score = dict()
        # rel_vec = rel_vec.transpose()
        logger.info('rel_vec shape: {}'.format(rel_vec.shape))
        for abs, sc in enumerate(rel_vec):
            sid2score[abs2sid[abs]] = sc

        sid_score_list = rank_sent.sort_sid2score(sid2score)

        out_fp = join(rank_dp, cid)
        logger.info('[rank_end2end] dumping ranking file to: {0}'.format(out_fp))
        rank_records = rank_sent.get_rank_records(sid_score_list)
        n_sents = rank_sent.dump_rank_records(rank_records, out_fp=out_fp)
        logger.info(
            '[rank_end2end] successfully dumped ranking of {0} sentences for {1}'.format(n_sents, cid))


def select_end2end(model_name, cos_threshold):
    select_sent.select_end2end(model_name=model_name,
                               n_iter=None,
                               sort='date',
                               cos_threshold=cos_threshold,
                               tokenize=False)
