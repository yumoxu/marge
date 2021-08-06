import io
import os
from os.path import isfile, isdir, join, dirname, abspath, exists
import sys
import json
import numpy as np
from utils.config_loader import logger, path_parser
import utils.tools as tools


sys.path.insert(0, dirname(dirname(abspath(__file__))))


def get_summ_comp_root(model_name, n_iter, mode):
    """

    :param model_name:
    :param attn_weigh:
    :param doc_weigh:
    :param n_iter:
    :param mode: w or r
    :return:
    """
    dn_items = tools.get_dir_name_items(model_name, n_iter)
    root_dp = join(path_parser.graph, '-'.join(dn_items))

    if mode == 'r':
        if not exists(root_dp):
            raise ValueError(f'root_dp does not exists: {root_dp}')
    elif mode == 'w':
        if exists(root_dp):
            raise ValueError(f'root_dp already exists: {root_dp}')
        os.mkdir(root_dp)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    return root_dp


def get_sim_mat_dp(summ_comp_root, mode):
    sim_mat_dp = join(summ_comp_root, 'sim_mat')

    if mode == 'r':
        if not exists(sim_mat_dp):
            raise ValueError('sim_mat_dp does not exists: {}'.format(sim_mat_dp))
    elif mode == 'w':
        if exists(sim_mat_dp):
            raise ValueError('sim_mat_dp already exists: {}'.format(sim_mat_dp))
        os.mkdir(sim_mat_dp)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    return sim_mat_dp


def get_rel_vec_dp(summ_comp_root, mode):
    rel_vec_dp = join(summ_comp_root, 'rel_vec')

    if mode == 'r':
        if not exists(rel_vec_dp):
            raise ValueError('rel_vec_dp does not exists: {}'.format(rel_vec_dp))
    elif mode == 'w':
        if exists(rel_vec_dp):
            raise ValueError('rel_vec_dp already exists: {}'.format(rel_vec_dp))
        os.mkdir(rel_vec_dp)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    return rel_vec_dp


def get_sid2abs_dp(summ_comp_root, mode):
    sid2abs_dp = join(summ_comp_root, 'sid2abs')

    if mode == 'r':
        if not exists(sid2abs_dp):
            raise ValueError('sid2abs_dp does not exists: {}'.format(sid2abs_dp))
    elif mode == 'w':
        if exists(sid2abs_dp):
            raise ValueError('sid2abs_dp already exists: {}'.format(sid2abs_dp))
        os.mkdir(sid2abs_dp)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    return sid2abs_dp


def get_sid2score_dp(summ_comp_root, mode):
    sid2score_dp = join(summ_comp_root, 'sid2score')

    if mode == 'r':
        if not exists(sid2score_dp):
            raise ValueError('sid2score_dp does not exists: {}'.format(sid2score_dp))
    elif mode == 'w':
        if exists(sid2score_dp):
            raise ValueError('sid2score_dp already exists: {}'.format(sid2score_dp))
        os.mkdir(sid2score_dp)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    return sid2score_dp


def dump_sim_mat(sim_mat, sim_mat_dp, cid):
    np.save(join(sim_mat_dp, cid), sim_mat)


def load_sim_mat(sim_mat_dp, cid):
    return np.load(join(sim_mat_dp, '{}.npy'.format(cid)))


def dump_rel_vec(rel_vec, rel_vec_dp, cid):
    np.save(join(rel_vec_dp, cid), rel_vec)


def load_rel_vec(rel_vec_dp, cid):
    return np.load(join(rel_vec_dp, '{}.npy'.format(cid)))


def dump_sid2abs(sid2abs, sid2abs_dp, cid):
    fp = '{}.json'.format(join(sid2abs_dp, cid))
    with io.open(fp, 'a') as f:
        json.dump(sid2abs, f)


def load_sid2abs(sid2abs_dp, cid):
    fp = '{}.json'.format(join(sid2abs_dp, cid))
    with io.open(fp, 'r') as f:
        return json.load(f)


def dump_sid2score(sid2score, sid2score_dp, cid):
    fp = '{}.json'.format(join(sid2score_dp, cid))
    with io.open(fp, 'a') as f:
        json.dump(sid2score, f)


def load_sid2score(sid2score_dp, cid):
    fp = '{}.json'.format(join(sid2score_dp, cid))
    with io.open(fp, 'r') as f:
        return json.load(f)


def load_components(sim_mat_dp, rel_vec_dp, sid2abs_dp, cid):
    sim_mat = load_sim_mat(sim_mat_dp, cid)
    rel_vec = load_rel_vec(rel_vec_dp, cid)
    sid2abs = load_sid2abs(sid2abs_dp, cid)

    components = {
        'sim_mat': sim_mat,
        'rel_vec': rel_vec,
        'sid2abs': sid2abs,
    }
    return components