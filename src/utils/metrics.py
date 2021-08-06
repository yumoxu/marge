import io
import numpy as np
from os import listdir
from copy import deepcopy
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import math
from os.path import isfile, join, dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from utils.config_loader import config_model, logger, path_parser

score_func = config_model['score_func']


def one_class_f1(doc_scores):
    """
    :param y_true: d_batch * 1
    :param doc_scores: d_batch * n_query
    :return:
    """
    d_batch = doc_scores.shape[0]

    max_score_vec = np.max(doc_scores, axis=1)
    logger.info('max_score_vec: {}'.format(max_score_vec.shape))
    y_pred = np.equal(doc_scores[:, 0], max_score_vec).astype(int).reshape(d_batch)

    y_true = np.ones_like(y_pred, dtype=int)
    # y_true = y_true.astype(int).reshape(d_batch)
    # logger.info('[binary_metrics] y_true: {0}, y_pred: {1}'.format(y_true, y_pred))

    # y_true_set = set(y_true.tolist())
    # y_pred_set = set(y_pred.tolist())
    # logger.info('y_true labels: {0}, y_pred labels: {1}'.format(y_true_set, y_pred_set))

    pp = sklearn.metrics.precision_score(y_true, y_pred)
    rr = sklearn.metrics.recall_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)

    res = {
        'f1': f1,
        'precision': pp,
        'recall': rr,
    }

    return res

def binary_metrics(y_true, y_pred, threshold=0.5):
    """
    :param y_true: d_batch * 1
    :param y_pred: d_batch * 1
    :return:
    """
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))

    d_batch = y_true.shape[0]
    y_pred = (y_pred > threshold).astype(int).reshape(d_batch)
    y_true = y_true.astype(int).reshape(d_batch)

    logger.info('[binary_metrics] y_true: {0}, y_pred: {1}'.format(y_true.shape, y_pred.shape))
    # logger.info('[binary_metrics] y_true: {0}, y_pred: {1}'.format(y_true, y_pred))

    # y_true_set = set(y_true.tolist())
    # y_pred_set = set(y_pred.tolist())
    # logger.info('y_true labels: {0}, y_pred labels: {1}'.format(y_true_set, y_pred_set))

    pp = sklearn.metrics.precision_score(y_true, y_pred)
    rr = sklearn.metrics.recall_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)

    res = {
        'rmse': rmse,
        'f1': f1,
        'precision': pp,
        'recall': rr,
    }

    return res


def precision(y_true, y_pred):
    """
        return a list of precision values.
    :param y_true: d_batch * n_doms
    :param y_pred: d_batch * n_doms
    :return:
    """
    precision_list = list()
    n_sample = len(y_true)
    for sample_idx in range(n_sample):
        precision_list.append(precision_score(y_true[sample_idx, :],  y_pred[sample_idx, :]))

    return precision_list


def recall(y_true, y_pred):
    """
        return a list of recall values.
    :param y_true:
    :param y_pred:
    :return:
    """
    recall_list = list()
    n_sample = len(y_true)
    for sample_idx in range(n_sample):
        recall_list.append(recall_score(y_true[sample_idx, :], y_pred[sample_idx, :]))

    return recall_list


def compute_example_based_f1(precision_list, recall_list):
    n_samples = len(precision_list)
    logger.info('n_samples: {0}'.format(n_samples))

    example_based_precision = sum(precision_list) / n_samples
    example_based_recall = sum(recall_list) / n_samples

    f1 = 2 * example_based_precision * example_based_recall / (example_based_precision + example_based_recall)

    return f1


def transform_test_input_2d(y_true_2d, y_pred_2d, des_sent_info):
    """
        for evaluating sentence-level domain detection.

    :param y_true_2d: d_batch * n_doms
    :param y_pred_2d: d_batch * n_doms
    :param des_sent_info: d_batch * [start_sent_idx, end_sent_idx, n_sents]
    :return: y_true_sents, y_pred_sents: (d_batch * n_sents) * n_doms
    """
    d_batch = len(y_pred_2d)
    y_true_sents = list()
    y_pred_sents = list()

    for s_idx in range(d_batch):
        start_sent_idx, end_sent_idx, n_sents = des_sent_info[s_idx]

        y_true_sents_items = np.tile(y_true_2d[s_idx, :], (n_sents, 1))
        y_true_sents.append(y_true_sents_items)

        y_pred_sents_items = np.tile(y_pred_2d[s_idx, :], (n_sents, 1))
        y_pred_sents.append(y_pred_sents_items)

        if y_true_sents_items.shape != y_pred_sents_items.shape:
            logger.error('des_sent_info: {0}'.format(des_sent_info[s_idx]))
            logger.error('y_true_sents_items: {0}'.format(y_true_sents_items))
            logger.error('y_pred_sents_items: {0}'.format(y_pred_sents_items))
            raise ValueError

    y_true_sents = np.concatenate(y_true_sents)
    y_pred_sents = np.concatenate(y_pred_sents)

    logger.info('y_true_sents: {0}, y_pred_sents: {1}'.format(y_true_sents.shape, y_pred_sents.shape))
    assert y_true_sents.shape == y_pred_sents.shape

    return y_true_sents, y_pred_sents


def transform_test_input_3d(y_true_2d, y_pred_3d, des_sent_info):
    """
        for evaluating sentence-level domain detection.

    :param y_true_2d: d_batch * n_doms
    :param y_pred_3d: d_batch * n_sents * n_doms
    :param des_sent_info: d_batch * [start_sent_idx, end_sent_idx, n_sents]
    :return: y_true_sents, y_pred_sents: (d_batch * n_sents) * n_doms
    """
    d_batch = len(y_pred_3d)
    y_pred_sents = list()
    y_true_sents = list()

    for s_idx in range(d_batch):

        start_sent_idx, end_sent_idx, n_sents = des_sent_info[s_idx]

        # logger.info('d_batch: {0}, s_idx: {1}, n_sents: {2}'.format(d_batch, s_idx, n_sents))
        y_true_sents_items = np.tile(y_true_2d[s_idx, :], (n_sents, 1))
        y_true_sents.append(y_true_sents_items)

        y_pred_sents_items = y_pred_3d[s_idx, start_sent_idx:end_sent_idx, :]

        y_pred_sents.append(y_pred_sents_items)

        if y_true_sents_items.shape != y_pred_sents_items.shape:
            logger.error('des_sent_info: {0}'.format(des_sent_info[s_idx]))
            logger.error('y_true_sents_items: {0}'.format(y_true_sents_items))
            logger.error('y_pred_sents_items: {0}'.format(y_pred_sents_items))
            raise ValueError

    y_true_sents = np.concatenate(y_true_sents)
    y_pred_sents = np.concatenate(y_pred_sents)

    logger.info('y_true_sents: {0}, y_pred_sents: {1}'.format(y_true_sents.shape, y_pred_sents.shape))
    assert y_true_sents.shape == y_pred_sents.shape

    return y_true_sents, y_pred_sents


def transform_test_input_2d_for_mturk(y_true_3d, y_pred_2d, n_sents):
    """
        for evaluating sentence-level domain detection.

    :param y_true_3d: d_batch * max_n_sents * n_doms
    :param y_pred_2d: d_batch * n_doms
    :param n_sents: d_batch * 1
    :return: y_true_sents, y_pred_sents: (d_batch * total_sents) * n_doms
    """
    d_batch = len(y_pred_2d)
    y_true_sents = list()
    y_pred_sents = list()
    # logger.info('n_sents: {}'.format(n_sents))

    for s_idx in range(d_batch):
        y_true_sents_items = y_true_3d[s_idx, 0:n_sents[s_idx, 0], :]
        y_true_sents.append(y_true_sents_items)

        y_pred_sents_items = np.tile(y_pred_2d[s_idx, :], (n_sents[s_idx, 0], 1))
        y_pred_sents.append(y_pred_sents_items)

        if y_true_sents_items.shape != y_pred_sents_items.shape:
            logger.error('n_sents: {0}'.format(n_sents[s_idx]))
            logger.error('y_true_sents_items: {0}'.format(y_true_sents_items))
            logger.error('y_pred_sents_items: {0}'.format(y_pred_sents_items))
            raise ValueError

    y_true_sents = np.concatenate(y_true_sents)
    y_pred_sents = np.concatenate(y_pred_sents)

    logger.info('y_true_sents: {0}, y_pred_sents: {1}'.format(y_true_sents.shape, y_pred_sents.shape))
    assert y_true_sents.shape == y_pred_sents.shape

    return y_true_sents, y_pred_sents


def transform_test_input_3d_for_mturk(y_true_3d, y_pred_3d, n_sents):
    """
        for evaluating sentence-level domain detection.

    :param y_true_3d: d_batch * max_n_sents * n_doms
    :param y_pred_3d: d_batch * max_n_sents * n_doms
    :param n_sents: d_batch * 1
    :return: y_true_sents, y_pred_sents: (d_batch * total_sents) * n_doms
    """
    d_batch = len(y_pred_3d)
    y_true_sents = list()
    y_pred_sents = list()

    for s_idx in range(d_batch):
        y_true_sents_items = y_true_3d[s_idx, 0:n_sents[s_idx, 0], :]
        y_true_sents.append(y_true_sents_items)

        y_pred_sents_items = y_pred_3d[s_idx, 0:n_sents[s_idx, 0], :]
        y_pred_sents.append(y_pred_sents_items)

        if y_true_sents_items.shape != y_pred_sents_items.shape:
            logger.error('n_sents: {0}'.format(n_sents[s_idx]))
            logger.error('y_true_sents_items: {0}'.format(y_true_sents_items))
            logger.error('y_pred_sents_items: {0}'.format(y_pred_sents_items))
            raise ValueError

    y_true_sents = np.concatenate(y_true_sents)
    y_pred_sents = np.concatenate(y_pred_sents)

    logger.info('y_true_sents: {0}, y_pred_sents: {1}'.format(y_true_sents.shape, y_pred_sents.shape))
    assert y_true_sents.shape == y_pred_sents.shape

    return y_true_sents, y_pred_sents


def get_y_pred_with_max_alter(doc_scores, threshold, zero_as_no):
    """
        If no hyp doc score is over the threshold,
        we choose the one with the highest score.
    """
    y_pred = deepcopy(doc_scores)
    y_pred[y_pred >= threshold] = 1.0

    pos_mask = (y_pred == 1.0)
    for sample_idx in range(y_pred.shape[0]):
        # print('pos_mask[sample_idx].sum().data: {}'.format(pos_mask[sample_idx].sum().data))
        if pos_mask[sample_idx].sum() == 0:  # no pos score
            hyp_max_idx = doc_scores[sample_idx].argmax()
            # logger.info('Use max score! set {0} to 1.0'.format(hyp_max_idx))
            y_pred[sample_idx][hyp_max_idx] = 1.0

    y_pred[y_pred < threshold] = 0.0 if zero_as_no else -1.0

    return y_pred


def get_y_pred_2d(scores_2d, max_alter):
    """
        get doc-level predictions with hyp scores.

    :param doc_scores: d_batch * n_doms
    :param max_alter: pick the best one when no one is above the threshold
    :return:
    """
    if score_func == 'softmax':
        for sample_idx in range(scores_2d.shape[0]):
            hyp_max_idx = scores_2d[sample_idx].argmax()
            scores_2d[sample_idx][hyp_max_idx] = 1.0
            scores_2d[sample_idx][:hyp_max_idx] = 0.0
            scores_2d[sample_idx][hyp_max_idx+1:] = 0.0
        return scores_2d

    if score_func == 'tanh':
        threshold = 0.0
        zero_as_no = False  # -1 as no
    else:
        threshold = 0.5
        zero_as_no = True

    if max_alter:  # choose the max idx as hyp when no hyp is made
        return get_y_pred_with_max_alter(scores_2d, threshold, zero_as_no)

    scores_2d[scores_2d >= threshold] = 1.0
    scores_2d[scores_2d < threshold] = 0.0 if zero_as_no else -1.0

    return scores_2d


def get_y_pred_3d(scores_3d, max_alter):
    """

    :param sent_scores: d_batch * n_sents * n_doms
    :param max_alter: pick the best one when no one is above the threshold
    :return:
    """
    threshold = 0.0 if score_func == 'tanh' else 0.5

    scores_3d[scores_3d >= threshold] = 1.0

    d_batch, n_sents = scores_3d.shape[0], scores_3d.shape[1]

    if max_alter:  # choose the max idx as hyp when no hyp is made
        pos_hyp_scores_bool = (scores_3d == 1.0)
        for sample_idx in range(d_batch):
            for sent_idx in range(n_sents):
                if pos_hyp_scores_bool[sample_idx, sent_idx].sum() == 0:
                    hyp_max_idx = scores_3d[sample_idx, sent_idx].argmax()
                    scores_3d[sample_idx, sent_idx][hyp_max_idx] = 1.0

    scores_3d[scores_3d < threshold] = -1.0 if score_func == 'tanh' else 0.0

    return scores_3d


def metric_eval(y_true, hyp_scores, des_sent_info=None, is_hiernet=False, save_pred_to=None, save_true_to=None):
    """
        Eval sent or doc predictions.

    :param y_true: d_batch * n_doms
    :param hyp_scores: batch * n_dom if des_sent_info else batch * n_sents * n_dom
    :param des_sent_info: doc evaluation if None else sent evaluation.
    :return: metric results
    """
    if des_sent_info is None or (des_sent_info is not None and is_hiernet):
        # # doc eval or sent eval for hiernet
        y_pred = get_y_pred_2d(hyp_scores, max_alter=True)
    else:  # sent eval for non-hiernet model
        y_pred = get_y_pred_3d(hyp_scores, max_alter=True)
        y_true, y_pred = transform_test_input_3d(y_true_2d=y_true, y_pred_3d=y_pred, des_sent_info=des_sent_info)

    if save_pred_to:
        save_numpy(y_pred, fp=save_pred_to)

    if save_true_to:
        save_numpy(y_true, fp=save_true_to)

    precision_list = precision(y_true, y_pred)
    recall_list = recall(y_true, y_pred)
    res = {
        'precision_list': precision_list,
        'recall_list': recall_list,
    }

    return res


def save_numpy(mat, fp):
    str_list = [' '.join([str(scalar) for scalar in vec]) for vec in mat.tolist()]

    # if ex_check and isfile(fp):
    #     logger.info('file exist: {0}'.format(fp))
    #     return

    with io.open(fp, mode='a', encoding='utf-8') as f:
        f.write('\n'.join(str_list) + '\n')
    logger.info('save to: {0}'.format(fp))


def metric_eval_for_mturk(y_true, hyp_scores, n_sents, is_hiernet=False, save_pred_to=None, save_true_to=None):
    """
        Eval sent predictions for mturk.

    :param y_true: batch * max_n_sents * n_doms
    :param hyp_scores: batch * n_doms if is_hiernet else batch * n_sents * n_dom
    :param n_sents: batch * 1

    :return: metric results
    """
    if is_hiernet:
        y_pred = get_y_pred_2d(hyp_scores, max_alter=True)
        y_true, y_pred = transform_test_input_2d_for_mturk(y_true_3d=y_true, y_pred_2d=y_pred, n_sents=n_sents)
    else:
        y_pred = get_y_pred_3d(hyp_scores, max_alter=True)
        y_true, y_pred = transform_test_input_3d_for_mturk(y_true_3d=y_true, y_pred_3d=y_pred, n_sents=n_sents)

    if save_pred_to:
        save_numpy(y_pred, fp=save_pred_to)

    if save_true_to:
        save_numpy(y_true, fp=save_true_to)

    precision_list = precision(y_true, y_pred)
    recall_list = recall(y_true, y_pred)

    res = {
        'precision_list': precision_list,
        'recall_list': recall_list,
    }

    return res


def metric_eval_with_y_pred(y_true, y_pred):
    precision_list = precision(y_true, y_pred)
    recall_list = recall(y_true, y_pred)

    res = {
        'precision_list': precision_list,
        'recall_list': recall_list,
    }

    return res


def transform_test_input_for_fabricated_doc(y_pred, fids, label_fns, tile_y_pred, zero_as_no=False):
    """

    :param y_pred: d_batch * n_sents * n_doms or d_batch * n_doms
    :param fids:
    :param label_fns:
    :param tile_y_pred: True for y_pred (d_batch * n_doms)
    :return:
    """
    y_pred_sents = list()
    y_true_sents = list()

    for s_idx, fid in enumerate(fids):
        y_true_sents_items = get_y_true_by_fid(label_fns, fid[0], zero_as_no)
        y_true_sents.append(y_true_sents_items)

        n_sents = len(y_true_sents_items)

        if tile_y_pred:
            y_pred_sents_items = np.tile(y_pred[s_idx, :], (n_sents, 1))
        else:
            y_pred_sents_items = y_pred[s_idx, :n_sents, :]

        y_pred_sents.append(y_pred_sents_items)

        if y_true_sents_items.shape != y_pred_sents_items.shape:
            logger.error('Shape: y_true_sents_items: {0}, y_pred_sents_items: {1}'.format(y_true_sents_items.shape,
                                                                                          y_pred_sents_items.shape))
            logger.error('y_true_sents_items: {0}'.format(y_true_sents_items))
            logger.error('y_pred_sents_items: {0}'.format(y_pred_sents_items))
            assert False
    y_true_sents = np.concatenate(y_true_sents)
    y_pred_sents = np.concatenate(y_pred_sents)

    logger.info('y_true_sents: {0}, y_pred_sents: {1}'.format(y_true_sents.shape, y_pred_sents.shape))
    assert y_true_sents.shape == y_pred_sents.shape

    return y_true_sents, y_pred_sents


def get_y_true_by_fid(label_fns, fid, zero_as_no=False):
    label_fn = None
    for fn in label_fns:
        if fn.startswith(str(fid) + '_'):
            label_fn = fn
            break

    if not label_fn:
        print('fid: {0}, label_fns: {1}'.format(fid, label_fns))
        assert label_fn

    with io.open(join(path_parser.dataset_fabricated_docs, label_fn), encoding='utf-8') as f:
        list_of_labels = [line.strip('\n').split('_') for line in f.readlines()]

        list_of_label_vecs = list()
        for labels in list_of_labels:
            if zero_as_no:
                label_vec = np.zeros([len(doms_final)], dtype=np.float32)
            else:
                label_vec = - np.ones([len(doms_final)], dtype=np.float32)

            for label in labels:
                label_vec[doms_final.index(label)] = 1.0

            list_of_label_vecs.append(label_vec)

    return np.stack(list_of_label_vecs)


def metric_eval_for_fabricated_doc(hyp_scores, fids, is_hiernet, zero_as_no=False, save_pred_to=None, save_true_to=None):
    """
    :param hyp_scores: d_batch * n_sents * n_doms or d_batch * n_doms (hiernet)
    :param fids: d_batch * 1
    :param is_hiernet: boolean.
    :return:
    """
    if is_hiernet:
        y_pred = get_y_pred_2d(scores_2d=hyp_scores, max_alter=True)
    else:
        y_pred = get_y_pred_3d(scores_3d=hyp_scores, max_alter=True)

    root = path_parser.dataset_fabricated_docs
    label_fns = [fn for fn in listdir(root) if fn.endswith('label') and isfile(join(root, fn))]

    y_true, y_pred = transform_test_input_for_fabricated_doc(y_pred, fids, label_fns,
                                                             tile_y_pred=is_hiernet,
                                                             zero_as_no=zero_as_no)

    if save_pred_to:
        save_numpy(y_pred, fp=save_pred_to)

    if save_true_to:
        save_numpy(y_true, fp=save_true_to)

    precision_list = precision(y_true, y_pred)
    recall_list = recall(y_true, y_pred)

    res = {
        'precision_list': precision_list,
        'recall_list': recall_list,
    }

    return res


def metric_eval_for_fabricated_doc_with_y_pred(y_pred, fids, tile_y_pred, zero_as_no):
    """
    :param y_pred: d_batch * n_sents * n_doms or d_batch * n_doms (hiernet)
    :param fids: d_batch * 1
    :param tile_y_pred: boolean.
    :return:
    """
    root = path_parser.dataset_fabricated_docs
    label_fns = [fn for fn in listdir(root) if fn.endswith('label') and isfile(join(root, fn))]

    y_true, y_pred = transform_test_input_for_fabricated_doc(y_pred, fids, label_fns,
                                                             tile_y_pred=tile_y_pred,
                                                             zero_as_no=zero_as_no)

    logger.info('y_true: {}'.format(y_true))
    logger.info('y_pred: {}'.format(y_pred))
    logger.info('y_true_size: {}'.format(y_true.shape))

    precision_list = precision(y_true, y_pred)
    recall_list = recall(y_true, y_pred)

    res = {
        'precision_list': precision_list,
        'recall_list': recall_list,
    }

    return res
