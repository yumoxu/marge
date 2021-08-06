# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.stats import entropy
import copy
from data.dataset_parser import dataset_parser
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from ir.ir_tools import load_retrieved_sentences, load_retrieved_passages
import gensim.summarization.bm25 as bm25
import sklearn
import nltk
import math

import utils.config_loader as config
from utils.config_loader import logger, path_parser
from lexrank import STOPWORDS, LexRank


def get_counts(sents):
    count_vec = CountVectorizer()
    # logger.info('sents: {}'.format(sents))
    counts = count_vec.fit_transform(sents)
    return counts


def get_tf_idf_mat(sents):
    counts = get_counts(sents)
    # logger.info('counts.shape: {}'.format(counts.shape))
    tf_idf_transformer = TfidfTransformer()
    tf_idf_mat = tf_idf_transformer.fit_transform(counts)
    # logger.info('tf_idf_mat.shape: {}'.format(tf_idf_mat.shape))

    return tf_idf_mat


def get_tf_mat(sents):
    counts = get_counts(sents)
    # logger.info('counts.shape: {}'.format(counts.shape))
    tf_transformer = TfidfTransformer(use_idf=False)
    tf_mat = tf_transformer.fit_transform(counts)
    # logger.info('tf_idf_mat.shape: {}'.format(tf_idf_mat.shape))
    return tf_mat


def build_cross_document_mask(processed_sents):
    """

    :param processed_sents: 2d lists, docs => sents
    :return:
    """
    n_doc = len(processed_sents)
    n_sents = [len(doc_sents) for doc_sents in processed_sents]
    total_n_sent = sum(n_sents)

    start = 0
    submasks = []
    for doc_idx in range(n_doc):
        doc_sents = processed_sents[doc_idx]
        n_doc_sents = len(doc_sents)
        mask = np.ones([n_doc_sents, total_n_sent], dtype=int)
        end = start + n_doc_sents
        mask[:, start:end] = 0
        start = end
        submasks.append(mask)

    mask = np.concatenate(submasks, axis=0)
    return mask


def _compute_rel_scores_tf(processed_sents, query):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    doc_sents.append(query)

    tf_mat = get_tf_mat(sents=doc_sents)

    doc_sent_mat = tf_mat[:-1]
    query_mat = tf_mat[-1]
    doc_query_sim_mat = cosine_similarity(doc_sent_mat, query_mat)

    rel_scores = np.squeeze(doc_query_sim_mat, axis=-1)

    return rel_scores


def _compute_rel_scores_tf_dot(processed_sents, query):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    # logger.info('doc_sents: {}'.format(len(doc_sents)))
    doc_sents.append(query)

    tf_mat = get_tf_mat(sents=doc_sents).toarray()
    # logger.info('tf_idf_mat: {}'.format(tf_idf_mat))

    # doc_sent_mat = tf_mat[:-1].A
    # query_mat = tf_mat[-1].A.reshape(-1, 1)
    # logger.info('doc_sent_mat: {}, query_mat: {}'.format(doc_sent_mat.shape, query_mat.shape))
    # logger.info('doc_sent_mat: {}'.format(type(doc_sent_mat)))
    rel_scores = np.matmul(tf_mat[:-1], tf_mat[-1])
    # rel_scores = np.squeeze(doc_query_sim_mat, axis=-1)

    logger.info('rel_scores: {}'.format(rel_scores.shape))

    return rel_scores


def _compute_rel_scores_count(processed_sents, query):
    """
    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    # todo: recheck and rerun
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_words = [nltk.tokenize.word_tokenize(sent) for sent in doc_sents]
    query_words = nltk.tokenize.word_tokenize(query)

    rel_scores = []
    for sent_words in doc_words:
        count = sum([sent_words.count(q_w) for q_w in query_words])
        rel_scores.append(count)

    rel_scores = np.array(rel_scores)

    return rel_scores


def _compute_rel_scores_bha(processed_sents, query):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_words = [nltk.tokenize.word_tokenize(sent) for sent in doc_sents]
    query_words = nltk.tokenize.word_tokenize(query)

    query_vocab = set(query_words)
    query_prob = {}

    for qw in query_vocab:
        query_prob[qw] = float(query_words.count(qw)) / len(query_words)

    rel_scores = []
    for sent_words in doc_words:
        mass = 0.0
        for qw in query_vocab:
            sentence_prob = float(sent_words.count(qw)) / len(sent_words)
            mass += math.sqrt(query_prob[qw] * sentence_prob)

        rel_scores.append(mass)

    rel_scores = np.array(rel_scores)
    return rel_scores


def _compute_rel_scores_weighted_count(processed_sents, query):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_words = [nltk.tokenize.word_tokenize(sent) for sent in doc_sents]
    query_words = nltk.tokenize.word_tokenize(query)

    query_vocab = set(query_words)
    query_prob = {}

    for qw in query_vocab:
        query_prob[qw] = float(query_words.count(qw)) / len(query_words)

    rel_scores = []
    for sent_words in doc_words:
        mass = 0.0
        for qw in query_vocab:
            mass += query_prob[qw] * float(sent_words.count(qw))

        rel_scores.append(mass)

    rel_scores = np.array(rel_scores)
    return rel_scores


def _compute_rel_scores_kl(processed_sents, query):
    """
        todo: this function is not finished.
              the entropy function needs to further handle zero denominator.

        :param processed_sents:
        :param query_sents:
        :param mask_intra:
        :return:
        """

    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    doc_sents.append(query)

    tf_mat = get_tf_mat(sents=doc_sents)
    # logger.info('tf_idf_mat: {}'.format(tf_idf_mat))

    denom = np.sum(tf_mat, axis=-1)
    distributions = tf_mat / denom

    sent_dists = distributions[:-1]
    query_dist = distributions[-1]

    rel_scores = []
    for sent_dist in sent_dists:
        score = entropy(sent_dist, query_dist)
        rel_scores.append(score)
    rel_scores = np.array(rel_scores)

    return rel_scores


def _compute_rel_scores_bm25(processed_sents, query):
    """

    :param doc_sents:  1d sent list (processed)
    :param query:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_words = [nltk.tokenize.word_tokenize(sent) for sent in doc_sents]
    query_words = nltk.tokenize.word_tokenize(query)

    corpus = copy.deepcopy(doc_words)
    corpus.append(query_words)
    bm_25_obj = bm25.BM25(corpus=corpus)

    scores = bm_25_obj.get_scores(document=query_words)[:-1]  # index:-1 is query itself

    if len(scores) != len(doc_sents):
        raise ValueError('Incompatible length between scores: {}, #sentences: {}'.format(len(scores), len(doc_sents)))

    rel_scores = np.array(scores)

    return rel_scores


def _compute_sim_mat_tfidf(processed_sents, query, mask_intra):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """

    # todo: this implementation has a bug: tfidf should be calculated between all sentences, instead of in a document.
    # logger.info('query: {}'.format(query))
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list

    # print('docs_1d: {}'.format(flat_processed_sents))
    # logger.info('doc_sents: {}'.format(doc_sents))
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    doc_sents.append(query)

    # logger.info('doc_sents: {}'.format(doc_sents))
    tf_idf_mat = get_tf_idf_mat(sents=doc_sents)
    # logger.info('tf_idf_mat: {}'.format(tf_idf_mat))

    doc_sent_mat = tf_idf_mat[:-1]
    query_mat = tf_idf_mat[-1]
    # logger.info('doc_sent_mat: {}'.format(doc_sent_mat.shape))
    # logger.info('query_sent_mat: {}'.format(query_sent_mat.shape))
    doc_query_sim_mat = cosine_similarity(doc_sent_mat, query_mat)
    doc_sim_mat = cosine_similarity(doc_sent_mat)

    # logger.info('[BEFORE] axis sum of doc_sim_mat: {}'.format(np.sum(doc_sim_mat, axis=0)))
    if mask_intra:
        inter_mask = build_cross_document_mask(processed_sents)
        doc_sim_mat *= inter_mask
        # doc_sim_mat = normalize(doc_sim_mat, axis=1, copy=True)
        # logger.info('[AFTER] axis sum of doc_sim_mat: {}'.format(np.sum(doc_sim_mat, axis=0)))

    rel_scores = np.squeeze(doc_query_sim_mat, axis=-1)

    res = {
        'rel_scores': rel_scores,  # todo: check shape
        'doc_sim_mat': doc_sim_mat,
    }

    return res


def _compute_sim_mat_bm25(processed_sents, query, mask_intra, conf):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    # compute doc_sim_mat
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_words = [nltk.tokenize.word_tokenize(sent) for sent in doc_sents]

    corpus = copy.deepcopy(doc_words)
    sim_scores = bm25.get_bm25_weights(corpus=corpus, n_jobs=-1)  # size: (n_sents + 1) * (n_sents + 1)
    sim_mat = np.array(sim_scores)  # need normalization

    corpus_size = len(corpus)
    if conf and conf < 1.0:
        sim_mat_normed = sklearn.preprocessing.normalize(sim_mat, axis=1, norm='l1')
        for row_idx in range(corpus_size):
            row = sim_mat_normed[row_idx]
            sorted_ids = np.argsort(row)[::-1]  # descending
            # logger.info('sorted ids: {}'.format(ids))

            for k in range(corpus_size):
                selected_ids = sorted_ids[:k + 1]
                # logger.info('selected_ids: {}, row[selected_ids]: {}'.format(selected_ids, row[selected_ids]))
                if sum(row[selected_ids]) < conf:
                    continue

                clip_ids = [j for j in range(corpus_size) if j not in selected_ids]
                sim_mat[row_idx][clip_ids] = 0.0
                print('k: {}/{}'.format(k, corpus_size))
                break

    # compute rel_scores
    query_words = nltk.tokenize.word_tokenize(query)
    corpus.append(query_words)
    bm_25_obj = bm25.BM25(corpus=corpus)
    rel_scores = bm_25_obj.get_scores(document=query_words)[:-1]  # index:-1 is query itself
    rel_scores = np.array(rel_scores)

    res = {
        'doc_sim_mat': sim_mat,
        'rel_scores': rel_scores,  # todo: check shape
    }

    return res


def _compute_sim_mat_partial_bm25(processed_sents, query, mask_intra):
    """

    :param processed_sents:
    :param query_sents:
    :param mask_intra:
    :return:
    """
    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list

    # print('docs_1d: {}'.format(flat_processed_sents))
    # logger.info('doc_sents: {}'.format(doc_sents))
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list

    tf_idf_mat = get_tf_idf_mat(sents=doc_sents)
    # logger.info('tf_idf_mat: {}'.format(tf_idf_mat))
    doc_sim_mat = cosine_similarity(tf_idf_mat)
    rel_scores = _compute_rel_scores_bm25(doc_sents, query)

    # logger.info('[BEFORE] axis sum of doc_sim_mat: {}'.format(np.sum(doc_sim_mat, axis=0)))
    if mask_intra:
        inter_mask = build_cross_document_mask(processed_sents)
        doc_sim_mat *= inter_mask
        # doc_sim_mat = normalize(doc_sim_mat, axis=1, copy=True)
        # logger.info('[AFTER] axis sum of doc_sim_mat: {}'.format(np.sum(doc_sim_mat, axis=0)))

    res = {
        'rel_scores': rel_scores,  # todo: check shape
        'doc_sim_mat': doc_sim_mat,
    }

    return res


def build_rel_scores_bm25(cid,
                          query,
                          max_ns_doc=None,
                          retrieved_dp=None):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp,
                                                                   cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents
    rel_scores = _compute_rel_scores_bm25(processed_sents, query)

    res = {
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return res


def build_rel_scores_tf(cid,
                        query,
                        rm_dialog,
                        max_ns_doc=None,
                        retrieved_dp=None):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp,
                                                                   cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid, rm_dialog=rm_dialog, max_ns_doc=max_ns_doc)  # 2d lists, docs => sents

    rel_scores = _compute_rel_scores_tf(processed_sents, query)
    res = {
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return res


def build_rel_scores_count(cid,
                           query,
                           max_ns_doc=None,
                           retrieved_dp=None):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp,
                                                                   cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents
    rel_scores = _compute_rel_scores_count(processed_sents, query)

    res = {
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return res


def build_rel_scores_bha(cid, query, max_ns_doc=None, retrieved_dp=None):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp,
                                                                   cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents
    rel_scores = _compute_rel_scores_bha(processed_sents, query)

    res = {
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return res


def build_rel_scores_weighted_count(cid,
                                    query,
                                    max_ns_doc=None,
                                    retrieved_dp=None):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp,
                                                                   cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents
    rel_scores = _compute_rel_scores_weighted_count(processed_sents, query)

    res = {
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return res


def build_rel_scores_tf_dot(cid,
                            query,
                            max_ns_doc=None,
                            retrieved_dp=None):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp,
                                                                   cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents
    rel_scores = _compute_rel_scores_tf_dot(processed_sents, query)

    res = {
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return res


def build_rel_scores_tf_passage(cid,
                                query,
                                retrieved_dp=None):
    _, proc_passages, passage_ids = load_retrieved_passages(cid,
                                                            get_sents=False,
                                                            retrieved_dp=retrieved_dp,
                                                            passage_ids=None)
    # passage_ids, passage_fps = get_passage_fps(cid, retrieved_dp=retrieved_dp)
    #     #
    #     # proc_passages = []
    # # passage_ids = []
    # for fp in passage_fps:
    #     # po = load_obj(fp)
    #     with open(fp, 'rb') as f:
    #         po = dill.load(f)
    #     # print('po: {}, type(po): {}'.format(po, type(po)))
    #     # passage_ids.append(po.pid)
    #
    #     passage = po.get_proc_passage()
    #     proc_passages.append(passage)
    #     # logger.info('{}: {}'.format(po.pid, passage))
    rel_scores = _compute_rel_scores_tf([proc_passages], query)  # nest proc_passages again for compatibility; todo: double check the nest level
    logger.info('rel_scores: {}'.format(rel_scores))

    pid2score = {}
    for pid, score in zip(passage_ids, rel_scores):
        pid2score[pid] = score

    return pid2score


def build_sim_items_e2e(cid,
                        query,
                        mask_intra,
                        max_ns_doc=None,
                        retrieved_dp=None,
                        sentence_rep='tfidf',
                        central_conf=None,
                        rm_dialog=True):
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   rm_dialog=rm_dialog,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents

    if sentence_rep == 'tfidf':
        res = _compute_sim_mat_tfidf(processed_sents=processed_sents,
                                     query=query,
                                     mask_intra=mask_intra)

    elif sentence_rep == 'bm25':
        res = _compute_sim_mat_bm25(processed_sents=processed_sents,
                                    query=query,
                                    mask_intra=mask_intra,
                                    conf=central_conf)
    else:
        raise ValueError('Invalid sentence_rep: {}'.format(sentence_rep))

    sim_items = {
        'doc_sim_mat': res['doc_sim_mat'],
        'rel_scores': res['rel_scores'],
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return sim_items


def build_sim_items_e2e_tfidf_with_lexrank(cid, query, max_ns_doc=None, retrieved_dp=None, rm_dialog=True):
    """
        Initialize LexRank with document-wise organized sentences to get true IDF.

    :param cid:
    :param query:
    :param max_ns_doc:
    :param retrieved_dp:
    :param rm_dialog:
    :return:
    """
    if retrieved_dp:
        original_sents, processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)
    else:
        original_sents, processed_sents = dataset_parser.cid2sents(cid,
                                                                   rm_dialog=rm_dialog,
                                                                   max_ns_doc=max_ns_doc)  # 2d lists, docs => sents

    lxr = LexRank(processed_sents, stopwords=STOPWORDS['en'])

    doc_sents = list(itertools.chain(*processed_sents))  # 1d sent list
    doc_sents = copy.deepcopy(doc_sents)  # avoid affecting the original doc_sents list
    doc_sents.append(query)

    sim_mat = lxr.get_tfidf_similarity_matrix(sentences=doc_sents)

    doc_sim_mat  = sim_mat[:-1, :-1]
    rel_scores = sim_mat[-1, :-1]
    logger.info('doc_sim_mat: {}, rel_scores: {}'.format(doc_sim_mat.shape, rel_scores.shape))

    sim_items = {
        'doc_sim_mat': doc_sim_mat,
        'rel_scores': rel_scores,
        'processed_sents': processed_sents,
        'original_sents': original_sents,
    }

    return sim_items

