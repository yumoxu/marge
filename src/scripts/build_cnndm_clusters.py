# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
import logging
import io

import json
from tqdm import tqdm

import re
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer

import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import utils.tools as tools
from data.cnndm_parser import CnnDmParser

if config.path_type == 'afs':
    DP_PROJ = Path('/disk/scratch/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_DATA = DP_PROJ / 'data'
DP_CLUSTER_CNNDM = DP_DATA / 'cnndm' / 'clusters'  # also has cnndm rank from DrQA

DATASET_VAR = 'train'  # specify this
rank_fn = f'{DATASET_VAR}-50.rank'
RANK_FP = DP_CLUSTER_CNNDM / rank_fn

COS_THRESHOLD = 0.6
DUMP_FN = f'cluster-{DATASET_VAR}-cos_{COS_THRESHOLD}.json'
DUMP_FP = DP_CLUSTER_CNNDM / DUMP_FN

CNNDM_DATASET_SIZE = {
    'train': 287227,
    'val': 13368,
    'test': 11490,
}
DATASET_VARS = list(CNNDM_DATASET_SIZE.keys())

porter_stemmer = PorterStemmer()

"""
    Build clusters for CNN/DM for generating long summaries.

    Pre-requisites: 
        - Rank file (from DrQA) where each CNN/DM sample treated as query

"""

def _load_cnndm(dataset_var):
    # get all (doc, summary) pairs
    parser = CnnDmParser(dataset_var=dataset_var)
    sample_generator = parser.sample_generator()
    id2doc, id2summary = {}, {}

    # load CNN/DM data into HashMaps
    for doc_idx, doc_sentences, summary in tqdm(sample_generator, total=CNNDM_DATASET_SIZE[dataset_var]):
        id2doc[str(doc_idx)] = doc_sentences
        id2summary[str(doc_idx)] = summary
    
    return id2doc, id2summary


def _load_rank_from_drqa(rank_fp):
    # load rank data into HashMaps
    id2rankIds = {}
    with io.open(rank_fp) as rank_f:
        for line in tqdm(rank_f):
            json_obj = json.loads(line)
            id2rankIds[json_obj['doc_id']] = json_obj['rank_ids']
    return id2rankIds


class Sample:
    def __init__(self, sample_id, doc, summary):
        self.sample_id = sample_id
        self.doc = doc
        self.summary = summary
        
        self.summary_words = nltk.tokenize.word_tokenize(summary)
        self.proc_summary_words =  self._get_proc_summary_words(summary)  # for measuring cos similarity

    def _get_proc_summary_words(self, summary, rm_stop=True, stem=True):
        summary = summary.lower()
        summary = re.sub(r'\s+', ' ', summary).strip()  # remove extra spaces

        if not summary:
            return []

        if rm_stop:
            summary = remove_stopwords(summary)

        if stem:
            summary = porter_stemmer.stem_sentence(summary)

        proc_words = nltk.tokenize.word_tokenize(summary)

        return proc_words

    
class ClusterBuilder:
    def __init__(self, main_sample, cand_samples, cos_threshold, target_summ_len=250):
        """
            Select from cand_samples for target_sample to form a cluster.
            
        """
        self.main_sample = main_sample
        self.cand_samples = cand_samples

        self.cos_threshold = cos_threshold
        self.target_summ_len = target_summ_len

        self.summary_words = [
            self.main_sample.proc_summary_words
        ]  # 2-d list organized by: summary => words
        self.cluster_samples = [self.main_sample]
        
    def _sim_cond(self, cand_sample):
        cand_summary_words = cand_sample.proc_summary_words

        if self.cos_threshold == 1.0:
            return True

        sims = (tools.compute_sent_cosine(cand_summary_words, _summary_words) 
                    for _summary_words in self.summary_words)
        
        if max(sims) < self.cos_threshold:
            return True

        return False

    def _select_samples(self):
        n_total_words = len(self.main_sample.summary_words)
        min_diff = abs(n_total_words - self.target_summ_len)
        
        n_breaks = 0
        MAX_BREAK = 5  # n trails to look for a sample with a reasonable #words 
        for cand_sample in self.cand_samples:
            if not self._sim_cond(cand_sample):
                continue
            
            n_cand_words = len(cand_sample.summary_words)  # add the genuine #words in original sent
            diff = abs(n_total_words + n_cand_words - self.target_summ_len)
            if diff > min_diff:
                n_breaks += 1
                if n_breaks == MAX_BREAK:
                    break
                continue
            
            n_total_words += n_cand_words
            min_diff = diff
            self.summary_words.append(cand_sample.proc_summary_words)
            self.cluster_samples.append(cand_sample)
        
        # print(f'n_total_words: {n_total_words}')

    def build(self):
        """
            Return a cluster object.
            
        """
        self._select_samples()
        cluster_docs, cluster_summaries, cluster_sample_ids = [], [], []
        for sample in self.cluster_samples:
            cluster_docs.append(sample.doc)
            cluster_summaries.append(sample.summary)
            cluster_sample_ids.append(sample.sample_id)

        cluster = {
            'cluster_id': self.main_sample.sample_id,
            'cluster_docs': cluster_docs,
            'cluster_summaries': cluster_summaries,
            'cluster_sample_ids': cluster_sample_ids
        }
        return cluster

    
def build_clusters(dataset_var, rank_fp, cos_threshold=0.6, target_summ_len=250):
    """
        Build clusters based on the TF-IDF rank from DrQA.
        
    """
    if dataset_var not in DATASET_VARS:
        raise ValueError(f'Illegal {dataset_var}!')

    if exists(DUMP_FP):
        raise ValueError(f'Remove {DUMP_FP} before dumping data')

    logger.info(f'Load CNN/DM docs and summaries: {dataset_var}')
    id2doc, id2summary = _load_cnndm(dataset_var)

    logger.info(f'Load CNN/DM rank: {dataset_var}')
    id2rankIds = _load_rank_from_drqa(rank_fp)
    
    with open(DUMP_FP, 'a') as f:
        for doc_id, rank_ids in tqdm(id2rankIds.items(), total=CNNDM_DATASET_SIZE[dataset_var]):
            main_sample = Sample(sample_id=doc_id, doc=id2doc[doc_id], summary=id2summary[doc_id])

            cand_samples = [Sample(sample_id=rid, doc=id2doc[rid], summary=id2summary[rid]) 
                for rid in rank_ids]
            cluster_builder = ClusterBuilder(main_sample, cand_samples, 
                cos_threshold, target_summ_len)
            cluster = cluster_builder.build()
            # dump cluster
            json_str = json.dumps(cluster, ensure_ascii=False)
            f.write(json_str+'\n')


def unit_test_build_clusters(dataset_var, rank_fp, cos_threshold=0.6, target_summ_len=250):
    """
        Build clusters based on the TF-IDF rank from DrQA.
        
    """
    id2doc = {
        "0": 'DOC 0 aa bb cc', 
        "1": 'DOC 1 dd ee ff', 
        "2": 'DOC 2 xx yy zz',
        "3": 'DOC 3 aa bb cc', 
    }
        
    id2summary = {
        "0": 'SUMMARY 0 abc', 
        "1": 'SUMMARY 1 def', 
        "2": 'SUMMARY 2 xyz',
        "3": 'SUMMARY 3 abc', 
    }


    id2rankIds = {
        "0": ["0", "3", "1", "2"],
        "1": ["1", "2", "0", "3"],
        "2": ["2", "1", "0", "3"],
        "3": ["3", "0", "1", "2"],
    }
    
    with open(DUMP_FP, 'a') as f:
        for doc_id, rank_ids in tqdm(id2rankIds.items()):
            samples = [Sample(sample_id=rid, doc=id2doc[rid], summary=id2summary[rid]) for rid in rank_ids]
            cluster_builder = ClusterBuilder(samples=samples, cos_threshold=cos_threshold, target_summ_len=250)
            cluster = cluster_builder.build()
            # dump cluster
            json_str = json.dumps(cluster, ensure_ascii=False)
            # f.write(json_str+'\n')
            print(json_str)


def cluster_stats():
    nc, nd, nw_summ = 0.0, 0.0, 0.0
    with open(DUMP_FP) as cluster_f:
        for line in tqdm(cluster_f, total=CNNDM_DATASET_SIZE['train']):
            nc += 1
            json_obj = json.loads(line)

            nd += len(json_obj['cluster_sample_ids'])

            summary = ' '.join(json_obj['cluster_summaries'])
            nw_summ += len(nltk.word_tokenize(summary))
    
    avg_nd = nd / nc
    avg_summ = nw_summ / nc
    print(f'#doc per cluster: {avg_nd}, #words per summary: {avg_summ}')


if __name__ == "__main__":
    # build_clusters(dataset_var=DATASET_VAR, rank_fp=RANK_FP)
    # unit_test_build_clusters(dataset_var=DATASET_VAR, rank_fp=RANK_FP)
    cluster_stats()
