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

import io
from tqdm import tqdm
from data.dataset_parser import dataset_parser
import utils.config_loader as config
from utils.config_loader import logger
import utils.tools as tools
import summ.compute_rouge as rouge
import nltk
from ir.ir_tools import load_retrieved_sentences

sys.path.insert(0, dirname(dirname(abspath(__file__))))


class Selector:
    def __init__(self,
                 cid,
                 rank_fp,
                 text_dp,
                 cos_threshold,
                 ngram_block,
                 max_n_summary_words,
                 rel_sents_dp=None,
                 retrieved_dp=None,
                 rm_dialog=True):
        """
            before generate summaries,
            rank sentences in a cluster first,
            and save the rankings (see model_exec.py).

        :param keep_order: binary; keep the original order of summary sentences or not.
        :param rm_dialog: only useful when retrieved_dp=None
        """
        self.cid = cid
        self.cos_threshold = cos_threshold
        self.ngram_block = ngram_block

        self.word_tokenize = nltk.tokenize.word_tokenize

        # fps for rank and text
        self.rank_fp = rank_fp
        if not exists(self.rank_fp):
            raise ValueError(f'rank_fp does not exist: {rank_fp}')

        self.text_fp = join(text_dp, cid)  # for dumping summaries

        # 2|3-d list organized by: docs => paragraphs => sents
        if rel_sents_dp and retrieved_dp:
            raise ValueError('Specify only one of rel_sents_dp and retrieved_dp!')

        if rel_sents_dp:
            self.use_filter_sents = True
            rel_sents_fp = join(rel_sents_dp, cid)
            self.original_sents, self.processed_sents = dataset_parser.parse_rel_sents_file(rel_sents_fp)  # 1d sentence lists

        elif retrieved_dp:
            self.use_filter_sents = False
            self.original_sents, self.processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)  # 1d sentence lists

        else:
            self.use_filter_sents = False
            self.original_sents, self.processed_sents = dataset_parser.cid2sents(cid, rm_dialog=rm_dialog)  # 2d sentence lists

        if max_n_summary_words:
            self.max_n_summary_words = max_n_summary_words

        logger.info(f'[Selector.__init__] max_nw for {cid}: {self.max_n_summary_words}')
        self.summary_sent_words = []  # 2-d list organized by: sents => words

    def _load_ranking(self):
        with io.open(self.rank_fp, encoding='utf-8') as f:
            content = f.readlines()

        ranked_sent_ids = [ll.rstrip('\n').split('\t')[0] for ll in content]
        return ranked_sent_ids

    def _get_ngrams(self, n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_n(self, n, cand_sent_words, summary_sent_words):
        tri_c = self._get_ngrams(n, cand_sent_words)
        for summary_words in summary_sent_words:
            tri_s = self._get_ngrams(n, summary_words)
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    def _sim_cond(self, cand_sent_words):
        if not self.summary_sent_words:
            # logger.info('[SELECT SENT] selecting the first sentence')
            return True

        if self.cos_threshold == 1.0:
            return True

        if self.ngram_block and self._block_n(n=self.ngram_block, cand_sent_words=cand_sent_words, summary_sent_words=self.summary_sent_words):
            return False

        sims = (tools.compute_sent_cosine(cand_sent_words, summary_words) for summary_words in self.summary_sent_words)
        if max(sims) < self.cos_threshold:
            return True

        return False

    def _get_sent(self, sents, sid):
        if '_' in sid:
            return tools.get_sent(sents, sid)
        else:
            return sents[int(sid)]
        # if self.use_filter_sents:
        #     return sents[int(sid)]
        # else:
        #     return tools.get_sent(sents, sid)

    def _select_sent_ids(self):
        ranked_sent_ids = self._load_ranking()
        # logger.info('ranked_sent_ids: {0}'.format(ranked_sent_ids))

        selected_sids = []
        n_total_words = 0

        for sid in ranked_sent_ids:
            cand_sent_original = self._get_sent(self.original_sents, sid)
            cand_sent_proc = self._get_sent(self.processed_sents, sid)

            if not cand_sent_proc:  # cand_sent_proc could be None
                continue
           
            cand_sent_words = self.word_tokenize(cand_sent_proc)
            
            if not self._sim_cond(cand_sent_words):
                continue

            self.summary_sent_words.append(cand_sent_words)

            selected_sids.append(sid)
            n_total_words += len(self.word_tokenize(cand_sent_original))  # add the genuine #words in original sent

            num_words_to_clip = n_total_words - self.max_n_summary_words
            if num_words_to_clip >= 0:
                # logger.info('stop because satisfying 250 words')
                break

        # logger.info('n_total_words: {}'.format(n_total_words))
        return selected_sids

    def _gen_summary_wo_tokenize(self):
        sids = self._select_sent_ids()
        # logger.info('sids: {}'.format(sids))
        wc = 0
        break_when_finish_flag = False
        selected_sents = []
        for sid in sids:
            cand_sent = self._get_sent(self.original_sents, sid).strip(' ')
            selected_sents.append(cand_sent)

            cand_words = self.word_tokenize(cand_sent)
            wc += len(cand_words)

            if break_when_finish_flag:
                break
            if wc >= self.max_n_summary_words:  # break in next iter to get one more additional sentence
                break_when_finish_flag = True

        summary = '\n'.join(selected_sents)
        return summary

    def gen_and_dump_summary(self):
        summary = self._gen_summary_wo_tokenize()
        with open(self.text_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(summary)



class SelectorSentenceBudget:
    def __init__(self,
                 cid,
                 rank_fp,
                 text_dp,
                 cos_threshold,
                 ngram_block,
                 max_n_summary_sentences,
                 rel_sents_dp=None,
                 retrieved_dp=None,
                 rm_dialog=True):
        """
            before generate summaries,
            rank sentences in a cluster first,
            and save the rankings (see model_exec.py).

        :param keep_order: binary; keep the original order of summary sentences or not.
        :param rm_dialog: only useful when retrieved_dp=None
        """
        self.cid = cid
        self.cos_threshold = cos_threshold
        self.ngram_block = ngram_block

        self.word_tokenize = nltk.tokenize.word_tokenize

        # fps for rank and text
        self.rank_fp = rank_fp
        if not exists(self.rank_fp):
            raise ValueError('rank_fp does not exist: {}'.format(self.rank_fp))

        self.text_fp = join(text_dp, cid)  # for dumping summaries

        # 2|3-d list organized by: docs => paragraphs => sents
        if rel_sents_dp and retrieved_dp:
            raise ValueError('Specify only one of rel_sents_dp and retrieved_dp!')

        if rel_sents_dp:
            self.use_filter_sents = True
            rel_sents_fp = join(rel_sents_dp, cid)
            self.original_sents, self.processed_sents = dataset_parser.parse_rel_sents_file(rel_sents_fp)  # 1d sentence lists

        elif retrieved_dp:
            self.use_filter_sents = False
            self.original_sents, self.processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)

        else:
            self.use_filter_sents = False
            if 'tdqfs' in config.test_year:
                self.original_sents, self.processed_sents = dataset_parser.cid2sents_tdqfs(cid)
            else:
                self.original_sents, self.processed_sents = dataset_parser.cid2sents(cid, rm_dialog=rm_dialog)

        self.max_n_summary_sentences = max_n_summary_sentences

        logger.info('[Selector.__init__] max_n_summary_sentences for {}: {}'.format(cid, self.max_n_summary_sentences))

        self.summary_sent_words = []  # 2-d list organized by: sents => words

    def _load_ranking(self):
        with io.open(self.rank_fp, encoding='utf-8') as f:
            content = f.readlines()

        ranked_sent_ids = [ll.rstrip('\n').split('\t')[0] for ll in content]
        return ranked_sent_ids

    def _get_ngrams(self, n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_n(self, n, cand_sent_words, summary_sent_words):
        tri_c = self._get_ngrams(n, cand_sent_words)
        for summary_words in summary_sent_words:
            tri_s = self._get_ngrams(n, summary_words)
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    def _sim_cond(self, cand_sent_words):
        if not self.summary_sent_words:
            # logger.info('[SELECT SENT] selecting the first sentence')
            return True

        if self.cos_threshold == 1.0:
            return True

        if self.ngram_block and self._block_n(n=self.ngram_block, cand_sent_words=cand_sent_words, summary_sent_words=self.summary_sent_words):
            return False

        sims = (tools.compute_sent_cosine(cand_sent_words, summary_words) for summary_words in self.summary_sent_words)
        if max(sims) < self.cos_threshold:
            return True

        return False

    def _get_sent(self, sents, sid):
        if '_' in sid:
            return tools.get_sent(sents, sid)
        else:
            return sents[int(sid)]

    def _select_sent_ids(self):
        ranked_sent_ids = self._load_ranking()
        selected_sids = []

        for sid in ranked_sent_ids:
            cand_sent_original = self._get_sent(self.original_sents, sid)
            cand_sent_proc = self._get_sent(self.processed_sents, sid)

            if not cand_sent_proc:  # cand_sent_proc could be None
                continue
           
            cand_sent_words = self.word_tokenize(cand_sent_proc)
            
            if not self._sim_cond(cand_sent_words):
                continue

            self.summary_sent_words.append(cand_sent_words)

            selected_sids.append(sid)

            if len(selected_sids) >= self.max_n_summary_sentences:
                break

        return selected_sids

    def _gen_summary_wo_tokenize(self):
        sids = self._select_sent_ids()
        break_when_finish_flag = False
        selected_sents = []
        for sid in sids:
            cand_sent = self._get_sent(self.original_sents, sid).strip(' ')
            selected_sents.append(cand_sent)

            if len(selected_sents) >= self.max_n_summary_sentences:
                break
        
        summary = '\n'.join(selected_sents)
        return summary

    def gen_and_dump_summary(self):
        summary = self._gen_summary_wo_tokenize()
        with open(self.text_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(summary)


class SelectorNaive:
    def __init__(self,
                 cid,
                 rank_fp,
                 text_dp,
                 cos_threshold,
                 max_n_summary_words):
        """
            before generate summaries,
            rank sentences in a cluster first,
            and save the rankings (see model_exec.py).
        """
        self.cid = cid
        self.cos_threshold = cos_threshold

        self.word_tokenize = nltk.tokenize.word_tokenize
        self.rank_fp = rank_fp
        if not exists(self.rank_fp):
            raise ValueError('rank_fp does not exist: {}'.format(self.rank_fp))

        self._load_ranking()

        self.text_fp = join(text_dp, cid)  # for dumping summaries

        self.max_n_summary_words = max_n_summary_words
        logger.info('[Selector.__init__] max_nw for {}: {}'.format(cid, self.max_n_summary_words))

        self.summary_sent_words = []  # 2-d list organized by: sents => words

    def _load_ranking(self):
        content = io.open(self.rank_fp, encoding='utf-8').readlines()
        self.sid2sent = {}
        self.ordered_sids = []
        for ll in content:
            items = ll.rstrip('\n').split('\t')
            sid = items[0]

            self.ordered_sids.append(sid)
            self.sid2sent[sid] = items[-1]

    def _sim_cond(self, cand_sent_words):
        if not self.summary_sent_words:
            return True

        if self.cos_threshold == 1.0:
            return True

        sims = (tools.compute_sent_cosine(cand_sent_words, summary_words) for summary_words in self.summary_sent_words)
        if max(sims) < self.cos_threshold:
            return True

        return False

    def _select_sent_ids(self):
        selected_sids = []
        n_total_words = 0

        for sid in self.ordered_sids:
            cand_sent_original = self.sid2sent[sid]
            cand_sent_proc = dataset_parser._proc_sent(cand_sent_original, rm_dialog=False, rm_stop=True, stem=True)
            cand_sent_words = self.word_tokenize(cand_sent_proc)

            if not self._sim_cond(cand_sent_words):
                continue

            self.summary_sent_words.append(cand_sent_words)

            selected_sids.append(sid)
            n_total_words += len(self.word_tokenize(cand_sent_original))  # add the genuine #words in original sent

            num_words_to_clip = n_total_words - self.max_n_summary_words
            if num_words_to_clip >= 0:
                break

        return selected_sids

    def gen_and_dump_summary(self):
        sids = self._select_sent_ids()
        # logger.info('sids: {}'.format(sids))
        wc = 0
        break_when_finish_flag = False
        selected_sents = []
        for sid in sids:
            cand_sent = self.sid2sent[sid]
            selected_sents.append(cand_sent)

            cand_words = self.word_tokenize(cand_sent)
            wc += len(cand_words)

            if break_when_finish_flag:
                break
            if wc >= self.max_n_summary_words:  # break in next iter to get one more additional sentence
                break_when_finish_flag = True

        summary = '\n'.join(selected_sents)
        with open(self.text_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(summary)


def select_end2end(model_name,
                   n_iter=None,
                   diversity_param_tuple=None,
                   ngram_block=None,
                   cos_threshold=None,
                   max_n_summary_words=500,
                   extra=None,
                   rank_model_name=None,
                   rel_sents_dp=None,
                   retrieved_dp=None,
                   rm_dialog=True,
                   cc_ids=None,
                   ):
    """

    :param model_name:
    :param n_iter:
    :param sort: date or origin
    :param duplication: if True, duplicated generated summary $n_refs$.
    :param attn_weigh: bool
    :param para_weigh: bool
    :param doc_weigh: bool
    :param cos_threshold: 0.5, 0.6
    :param max_n_summary_words: 500
    :param rank_model_name: you can specify rank_model_name; default is set to model_name.
    :param rm_dialog: only useful when retrieved_dp=None
    :param cc_ids: specify for mn dataset
    :return:
    """
    # make dump dir
    text_params = {
        'model_name': model_name,
        'cos_threshold': cos_threshold if not ngram_block else cos_threshold+ngram_block,
        'n_iter': n_iter,
        'diversity_param_tuple': diversity_param_tuple,
        'extra': extra,
    }

    text_dp = tools.init_text_dp(**text_params, budget=None)
    # date_sorted_doc_indices

    if not cc_ids:
        cc_ids = tools.get_test_cc_ids()

    base_selector_params = {
        'text_dp': text_dp,
        'ngram_block': ngram_block,
        'cos_threshold': cos_threshold,
        'max_n_summary_words': max_n_summary_words,
        'rel_sents_dp': rel_sents_dp,
        'retrieved_dp': retrieved_dp,
    }

    # logger.info('[SELECT SENTS] selecting sents for {} clusters'.format(len(cc_ids)))
    if not rank_model_name:
        rank_model_name = model_name

    rank_dp = tools.get_rank_dp(rank_model_name,
                                n_iter=n_iter,
                                diversity_param_tuple=diversity_param_tuple,
                                extra=extra)

    for cid in tqdm(cc_ids):
        rank_fp = join(rank_dp, cid)
        selector_params = {
            **base_selector_params,
            'cid': cid,
            'rank_fp': rank_fp,
            'rm_dialog': rm_dialog,
        }

        selector = Selector(**selector_params)
        selector.gen_and_dump_summary()

    logger.info('[SELECT SENT] successfully dumped selected sentences to: {}'.format(text_dp))

    output = rouge.compute_rouge(**text_params)
    return output


def select_for_dev(rank_dp,
                   text_dp,
                   ngram_block=None,
                   cos_threshold=None,
                   rel_sents_dp=None,
                   retrieved_dp=None
                   ):
    """
        For final tuning.

        Rouge-2 are evaluated with "-l 250"
    :return:
    """
    # make dump dir
    cc_ids = tools.get_test_cc_ids()

    base_selector_params = {
        'text_dp': text_dp,
        'ngram_block': ngram_block,
        'cos_threshold': cos_threshold if not ngram_block else cos_threshold+ngram_block,
        'max_n_summary_words': 500,
        'rel_sents_dp': rel_sents_dp,
        'retrieved_dp': retrieved_dp,
    }

    for cid in tqdm(cc_ids):
        rank_fp = join(rank_dp, cid)
        selector_params = {
            **base_selector_params,
            'cid': cid,
            'rank_fp': rank_fp,
        }

        selector = Selector(**selector_params)
        selector.gen_and_dump_summary()

    logger.info('[SELECT SENT] successfully dumped selected sentences to: {}'.format(text_dp))
    output = rouge.compute_rouge_for_dev(text_dp, tune_centrality=True)
    return output


def select_end2end_mn(model_name,
        cc_ids,
        n_iter=None,
        diversity_param_tuple=None,
        ngram_block=None,
        cos_threshold=None,
        max_n_summary_words=1000,
        max_n_summary_sentences=9,
        budget=None,
        extra=None,
        rank_model_name=None,
        rel_sents_dp=None,
        retrieved_dp=None,
        rm_dialog=True
        ):
    """

    :param model_name:
    :param n_iter:
    :param sort: date or origin
    :param duplication: if True, duplicated generated summary $n_refs$.
    :param attn_weigh: bool
    :param para_weigh: bool
    :param doc_weigh: bool
    :param cos_threshold: 0.5, 0.6
    :param max_n_summary_words: 500
    :param rank_model_name: you can specify rank_model_name; default is set to model_name.
    :param rm_dialog: only useful when retrieved_dp=None
    :return:
    """
    # make dump dir
    text_params = {
        'model_name': model_name,
        'cos_threshold': cos_threshold if not ngram_block else cos_threshold+ngram_block,
        'n_iter': n_iter,
        'diversity_param_tuple': diversity_param_tuple,
        'budget': budget,
        'extra': extra,
    }

    text_dp = tools.init_text_dp(**text_params)

    base_selector_params = {
        'text_dp': text_dp,
        'ngram_block': ngram_block,
        'cos_threshold': cos_threshold,
        'rel_sents_dp': rel_sents_dp,
        'retrieved_dp': retrieved_dp,
    }

    if max_n_summary_sentences:
        SELECTOR_CLASS = SelectorSentenceBudget
        base_selector_params['max_n_summary_sentences'] = max_n_summary_sentences
    elif max_n_summary_sentences:
        SELECTOR_CLASS = Selector
        base_selector_params['max_n_summary_words'] = max_n_summary_words
    else:
        raise ValueError('Set one budget!')

    # logger.info('[SELECT SENTS] selecting sents for {} clusters'.format(len(cc_ids)))
    if not rank_model_name:
        rank_model_name = model_name

    rank_dp = tools.get_rank_dp(rank_model_name,
                                n_iter=n_iter,
                                diversity_param_tuple=diversity_param_tuple,
                                extra=extra)

    for cid in tqdm(cc_ids):
        rank_fp = join(rank_dp, cid)
        selector_params = {
            **base_selector_params,
            'cid': cid,
            'rank_fp': rank_fp,
            'rm_dialog': rm_dialog,
        }

        selector = SELECTOR_CLASS(**selector_params)
        selector.gen_and_dump_summary()

    logger.info('[SELECT SENT] successfully dumped selected sentences to: {}'.format(text_dp))

    output = rouge.compute_rouge_for_mn(text_dp)
    return output


def select_for_ablation_study(model_name,
                              cos_threshold,
                              ):
    """
        For ablation study.

        Typically for evaluation of model w/o centrality module.

    :return:
    """
    text_params = {
        'model_name': model_name,
        'cos_threshold': cos_threshold,
        'n_iter': None,
        'diversity_param_tuple': None,
        'extra': None,
    }
    text_dp = tools.init_text_dp(**text_params)
    rank_dp = tools.get_rank_dp(model_name)

    base_selector_params = {
        'text_dp': text_dp,
        'cos_threshold': cos_threshold,
        'max_n_summary_words': 500,
    }

    cc_ids = tools.get_test_cc_ids()
    for cid in tqdm(cc_ids):
        rank_fp = join(rank_dp, cid)
        selector_params = {
            **base_selector_params,
            'cid': cid,
            'rank_fp': rank_fp,
        }

        selector = SelectorNaive(**selector_params)
        selector.gen_and_dump_summary()

    logger.info('[SELECT SENT] successfully dumped selected sentences to: {}'.format(text_dp))
    output = rouge.compute_rouge_for_ablation_study(text_dp)
    return output



class SelectorforELI5:
    def __init__(self,
                 cid,
                 rank_fp,
                 text_dp,
                 cos_threshold,
                 max_n_summary_sentences,
                 rel_sents_dp=None,
                 retrieved_dp=None,
                 rm_dialog=True):
        """
            before generate summaries,
            rank sentences in a cluster first,
            and save the rankings (see model_exec.py).

        :param keep_order: binary; keep the original order of summary sentences or not.
        :param rm_dialog: only useful when retrieved_dp=None
        """
        self.cid = cid
        self.cos_threshold = cos_threshold

        self.word_tokenize = nltk.tokenize.word_tokenize

        # fps for rank and text
        self.rank_fp = rank_fp
        if not exists(self.rank_fp):
            raise ValueError('rank_fp does not exist: {}'.format(self.rank_fp))

        self.text_fp = join(text_dp, cid)  # for dumping summaries

        # 2|3-d list organized by: docs => paragraphs => sents
        if rel_sents_dp and retrieved_dp:
            raise ValueError('Specify only one of rel_sents_dp and retrieved_dp!')

        if rel_sents_dp:
            self.use_filter_sents = True
            rel_sents_fp = join(rel_sents_dp, cid)
            self.original_sents, self.processed_sents = dataset_parser.parse_rel_sents_file(
                rel_sents_fp)  # 1d sentence lists

        elif retrieved_dp:
            self.use_filter_sents = False
            self.original_sents, self.processed_sents = load_retrieved_sentences(retrieved_dp=retrieved_dp, cid=cid)

        else:
            self.use_filter_sents = False
            self.original_sents, self.processed_sents = dataset_parser.cid2sents(cid, rm_dialog=rm_dialog)

        if max_n_summary_sentences:
            self.max_n_summary_sentences = max_n_summary_sentences

        self.summary_sent_words = []  # 2-d list organized by: sents => words

    def _load_ranking(self):
        with io.open(self.rank_fp, encoding='utf-8') as f:
            content = f.readlines()

        ranked_sent_ids = [ll.rstrip('\n').split('\t')[0] for ll in content]
        return ranked_sent_ids

    def _sim_cond(self, cand_sent_words):
        if not self.summary_sent_words:
            return True

        if self.cos_threshold == 1.0:
            return True

        sims = (tools.compute_sent_cosine(cand_sent_words, summary_words) for summary_words in self.summary_sent_words)
        if max(sims) < self.cos_threshold:
            return True

        return False

    def _get_sent(self, sents, sid):
        if '_' in sid:
            return tools.get_sent(sents, sid)
        else:
            return sents[int(sid)]

    def _select_sent_ids(self):
        ranked_sent_ids = self._load_ranking()
        budget = len(ranked_sent_ids) - self.max_n_summary_sentences
        if budget <= 0:
            return ranked_sent_ids
        
        selected_sids = []
        for idx, sid in enumerate(ranked_sent_ids):
            if budget <= 0:
                selected_sids.extend(ranked_sent_ids[idx:])
                return selected_sids
            
            cand_sent_original = self._get_sent(self.original_sents, sid)
            cand_sent_proc = self._get_sent(self.processed_sents, sid)
            cand_sent_words = self.word_tokenize(cand_sent_proc)
            
            if not self._sim_cond(cand_sent_words):
                budget -= 1
                continue

            self.summary_sent_words.append(cand_sent_words)
            selected_sids.append(sid)
        
        return selected_sids

    def _gen_summary_wo_tokenize(self):
        sids = self._select_sent_ids()
        # logger.info('sids: {}'.format(sids))
        # wc = 0
        selected_sents = []
        for sid in sids:
            cand_sent = self._get_sent(self.original_sents, sid).strip(' ')
            selected_sents.append(cand_sent)
            # cand_words = self.word_tokenize(cand_sent)
            # wc += len(cand_words)
            if len(selected_sents) == self.max_n_summary_sentences:
                break

        summary = '\n'.join(selected_sents)
        return summary

    def gen_and_dump_summary(self):
        summary = self._gen_summary_wo_tokenize()
        with open(self.text_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write(summary)


def select_end2end_for_eli5(model_name,
        n_iter=None,
        length_budget_tuple=None,
        diversity_param_tuple=None,
        cos_threshold=None,
        extra=None,
        rank_model_name=None,
        rel_sents_dp=None,
        retrieved_dp=None,
        rm_dialog=True,
        cc_ids=None
        ):
    """

    :param model_name:
    :param n_iter:
    :param cos_threshold: 0.5, 0.6
    :param max_n_summary_words: 500
    :param rank_model_name: you can specify rank_model_name; default is set to model_name.
    :param rm_dialog: only useful when retrieved_dp=None
    :return:
    """
    # make dump dir
    text_params = {
        'model_name': model_name,
        'cos_threshold': cos_threshold,
        'n_iter': n_iter,
        'diversity_param_tuple': diversity_param_tuple,
        'length_budget_tuple': length_budget_tuple,
        'extra': extra,
    }

    text_dp = tools.init_text_dp_for_eli5(**text_params)

    base_selector_params = {
        'text_dp': text_dp,
        'cos_threshold': cos_threshold,
        'rel_sents_dp': rel_sents_dp,
        'retrieved_dp': retrieved_dp,
    }

    budget_type, budget = length_budget_tuple
    if budget_type == 'ns':
        SelectorCls = SelectorforELI5
        base_selector_params['max_n_summary_sentences'] = budget 
    elif budget_type == 'nw':
        SelectorCls = Selector  
        base_selector_params['max_n_summary_words'] = 500  # make sure the budget is satisified but only first 250 will be evaluated
        base_selector_params['ngram_block'] = False
    else:
        raise ValueError(f'Invalid budget_type: {budget_type}')

    # logger.info('[SELECT SENTS] selecting sents for {} clusters'.format(len(cc_ids)))
    if not rank_model_name:
        rank_model_name = model_name

    rank_dp = tools.get_rank_dp(rank_model_name,
                                n_iter=n_iter,
                                diversity_param_tuple=diversity_param_tuple,
                                extra=extra)

    for cid in tqdm(cc_ids):
        rank_fp = join(rank_dp, cid)
        selector_params = {
            **base_selector_params,
            'cid': cid,
            'rank_fp': rank_fp,
            'rm_dialog': rm_dialog,
        }

        selector = SelectorCls(**selector_params)
        selector.gen_and_dump_summary()

    logger.info('[SELECT SENT] successfully dumped selected sentences to: {}'.format(text_dp))
