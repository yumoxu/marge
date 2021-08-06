from __future__ import absolute_import, division, print_function

from os import listdir
from os.path import exists, join, dirname, abspath
import shutil
import logging

logger = logging.getLogger(__name__)

def get_max_v(d):
    if d.values():
        return max(d.values())
    return 0.0


def get_min_kv(d, init_v=100):
    min_k = None
    min_v = init_v
    for k, v in d.items():
        if v < min_v:
            min_v = v
            min_k = k
    return min_k, min_v


def update_checkpoint_dict(checkpoint_dict, k, v, max_n_checkpoint=3):
    """

    :param checkpoint_dict:
    :param k:
    :param v: the larger, the better.
    :param max_n_checkpoint:
    :return:
    """
    update = False
    is_best = v > get_max_v(checkpoint_dict)

    if len(checkpoint_dict) < max_n_checkpoint:  # not full
        checkpoint_dict[k] = v
        update = True
        return checkpoint_dict, update, is_best

    min_k, min_v = get_min_kv(checkpoint_dict)
    if v > min_v:  # replace
        checkpoint_dict.pop(min_k)
        checkpoint_dict[k] = v
        update = True
        return checkpoint_dict, update, is_best

    return checkpoint_dict, update, is_best


def clean_outdated_checkpoints(checkpoint, checkpoint_dict):
    ckpt_dirs = [dir for dir in listdir(checkpoint) if dir.startswith('checkpoint-')]
    target_ckpt_dirs = ['checkpoint-{}'.format(n_iter) for n_iter in checkpoint_dict]

    for ckpt in ckpt_dirs:
        if ckpt not in target_ckpt_dirs:
            shutil.rmtree(join(checkpoint, ckpt))
            # os.remove(join(checkpoint, ckpt))
            logger.info('Remove checkpoint: {}'.format(ckpt))

    ckpt_dirs = [dir for dir in listdir(checkpoint) if dir.startswith('checkpoint-')]
    logger.info('Available #checkpoints: {}'.format(len(ckpt_dirs)))
    