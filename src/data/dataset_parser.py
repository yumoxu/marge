# -*- coding: utf-8 -*-
import io
from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import re
from os import listdir
from os.path import join, isfile, isdir
import itertools

import utils.config_loader as config
from utils.config_loader import logger, path_parser, config_meta, config_model
import utils.tools as tools

import data.clip_and_mask as cm
import data.clip_and_mask_sl as cm_sl

import nltk
from nltk.tokenize import sent_tokenize, TextTilingTokenizer

from tqdm import tqdm

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer

"""

    This class provide following information extraction functions:

    (1) get_doc:
        Get an article from file.
        Return natural paragraphs/subtopic tiles/whole artilce.

    (2) doc2sents:
        Based on func:get_doc, get sentences from an article.
        Optionally, sentences can be organized by paragraphs.

    (3) cid2sents:
        Based on func:doc2sents, get sentences from a cluster.

    This class also provide following parsing functions:

    (1) parse_doc2paras:
        Based on func:get_doc, parse a doc => a dict with keys: ('paras',
                                                                 'article_mask')

    (2) parse_doc2sents:
        Based on func:doc2sents,

"""

class DatasetParser:
    def __init__(self):
        # info
        self.cluster_info = dict()
        self.article_info = dict()

        # config
        # self.max_n_para_words = config_model['max_n_para_words']
        # self.max_n_query_words = config_model['max_n_query_words']

        # logger.info('[DATA PARSER] loading tokenizer')
        if config_meta['word_tokenizer'] == 'bert':
            self.word_tokenize = config.bert_tokenizer.tokenize
        elif config_meta['word_tokenizer'] == 'nltk':
            self.word_tokenize = nltk.tokenize.word_tokenize
        else:
            raise ValueError('Invalid word_tokenizer: {}'.format(config_meta['word_tokenizer']))

        self.sent_tokenize = nltk.tokenize.sent_tokenize
        self.porter_stemmer = PorterStemmer()

        # base pat
        BASE_PAT = '(?<=<{0}> )[\s\S]*?(?= </{0}>)'
        BASE_PAT_WITH_NEW_LINE = '(?<=<{0}>\n)[\s\S]*?(?=\n</{0}>)'
        BASE_PAT_WITH_RIGHT_NEW_LINE = '(?<=<{0}>)[\s\S]*?(?=\n</{0}>)'

        # query pat
        self.id_pat = re.compile(BASE_PAT.format('num'))
        self.title_pat = re.compile(BASE_PAT.format('title'))
        self.narr_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('narr'))

        # article pat
        self.text_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('TEXT'))
        self.graphic_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('GRAPHIC'))
        self.type_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('TYPE'))
        self.para_pat = re.compile(BASE_PAT_WITH_NEW_LINE.format('P'))

        # headline pat
        self.headline_pat_0 = re.compile(BASE_PAT.format('HEADLINE'))
        self.headline_pat_1 = re.compile(BASE_PAT_WITH_NEW_LINE.format('HEADLINE'))

        self.BACKUP_HEADLINE_PAT = '(?<=</{0}>)[\s\S]*?(?=<TEXT>)'
        self.headline_pat_2 = re.compile(self.BACKUP_HEADLINE_PAT.format('HEADLINE'))
        self.headline_pat_3 = re.compile(self.BACKUP_HEADLINE_PAT.format('SLUG'))
        self.slug_pat = re.compile(BASE_PAT.format('SLUG'))

        # date pat
        self.date_pat_0 = re.compile(BASE_PAT_WITH_RIGHT_NEW_LINE.format('DATE'))
        self.date_pat_1 = re.compile(BASE_PAT_WITH_NEW_LINE.format('DATE'))

        self.proc_params_for_questions = {
            'rm_dialog': False,
            'rm_stop': False,
            'stem': True,
        }

    def _get_word_ids(self, words):
        word_ids = config.bert_tokenizer.convert_tokens_to_ids(words)
        return word_ids

    def _proc_sent(self, sent, rm_dialog, rm_stop, stem, rm_short=None, min_nw_sent=3):
        sent = sent.lower()
        sent = re.sub(r'\s+', ' ', sent).strip()  # remove extra spaces

        if not sent:
            return None

        if rm_short and len(nltk.tokenize.word_tokenize(sent)) < min_nw_sent:
            return None

        if rm_dialog:
            # dialog_tokens = ["''", "``", '"']
            dialog_tokens = ["''", "``"]
            for tk in dialog_tokens:
                if tk in sent:
                    logger.info('Remove dialog')
                    return None

            # in DUC 2005, articles use ' instead of ''
            if config.test_year == '2005': # todo: make more precise rule for dialog
                if sent[0] == "'" and ('says' in sent or 'said' in sent):
                    logger.info('Remove dialog')
                    return None

        if rm_stop:
            sent = remove_stopwords(sent)

        if stem:
            sent = self.porter_stemmer.stem_sentence(sent)

        return sent

    def _proc_para(self, pp, rm_dialog=True, rm_stop=True, stem=True, to_str=False):
        """
            Return both original paragraph and processed paragraph.

        :param pp:
        :param rm_dialog:
        :param rm_stop:
        :param stem:
        :param to_str: if True, concatenate sentences and return.
        :return:
        """
        original_para_sents, processed_para_sents = [], []

        for ss in self.sent_tokenize(pp):
            ss_origin = self._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
            ss_proc = self._proc_sent(ss, rm_dialog=rm_dialog, rm_stop=rm_stop, stem=stem)

            if ss_proc:  # make sure the sent is not removed, i.e., is not empty and is not in a dialog
                original_para_sents.append(ss_origin)
                processed_para_sents.append(ss_proc)

        if not to_str:
            return original_para_sents, processed_para_sents

        para_origin = ' '.join(original_para_sents)
        para_proc = ' '.join(processed_para_sents)
        return para_origin, para_proc

    def get_doc(self, fp, concat_paras):
        """
            get an article from file.

            first get all natural paragraphs in the text, then:
                if concat_paras, return paragraphs joint by \n; else return paragraphs.
        """
        with io.open(fp, encoding='utf-8') as f:
            article = f.read()

        pats = [self.text_pat, self.graphic_pat]

        PARA_SEP = '\n\n'

        for pat in pats:
            text = re.search(pat, article)

            if not text:
                continue

            text = text.group()

            # if there is '<p>' in text, gather them to text
            paras = re.findall(self.para_pat, text)
            if paras:
                text = PARA_SEP.join(paras)

            if concat_paras:
                return text

            # for text tiling: if paragraph break is a single '\n', double it
            pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
            matches = pattern.finditer(text)
            if not matches:
                text.replace('\n', PARA_SEP)

            if paras:
                return paras
            else:
                return text.split(PARA_SEP)

        logger.warning('No article content in {0}'.format(fp))
        return None

    def doc2sents(self, fp, para_org=False, rm_dialog=True, rm_stop=True, stem=True, rm_short=None):
        """
        :param fp:
        :param para_org: bool

        :return:
            if para_org=True, 2-layer nested lists;
            else: flat lists.

        """
        paras = self.get_doc(fp, concat_paras=False)

        original_sents, processed_sents = [], []

        if not paras:
            return [], []

        for pp in paras:
            # original_para_sents, processed_para_sents = [], []
            #
            # for ss in self.sent_tokenize(pp):
            #     ss_origin = self._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
            #     ss_proc = self._proc_sent(ss, rm_dialog=rm_dialog, rm_stop=rm_stop, stem=stem)
            #
            #     if ss_proc:  # make sure the sent is not removed, i.e., is not empty and is not in a dialog
            #         original_para_sents.append(ss_origin)
            #         processed_para_sents.append(ss_proc)

            original_para_sents, processed_para_sents = self._proc_para(pp, rm_dialog=rm_dialog,
                                                                        rm_stop=rm_stop, stem=stem)

            if para_org:
                original_sents.append(original_para_sents)
                processed_sents.append(processed_para_sents)
            else:
                original_sents.extend(original_para_sents)
                processed_sents.extend(processed_para_sents)

        return original_sents, processed_sents

    def doc2paras(self, fp, rm_dialog=True, rm_stop=True, stem=True):
        paras = self.get_doc(fp, concat_paras=False)

        if not paras:
            return [], []

        original_paras, processed_paras = [], []
        for pp in paras:
            original_para_sents, processed_para_sents = self._proc_para(pp, rm_dialog=rm_dialog,
                                                                        rm_stop=rm_stop, stem=stem)

            para_origin = ' '.join(original_para_sents)
            para_proc = ' '.join(processed_para_sents)

            original_paras.append(para_origin)
            processed_paras.append(para_proc)

        return original_paras, processed_paras

    def cid2sents(self, cid, rm_dialog=True, rm_stop=True, stem=True, max_ns_doc=None):
        """
            Load all sentences in a cluster.

        :param cid:
        :param rm_dialog:
        :param rm_stop:
        :param stem:
        :param max_ns_doc:
        :return: a 2D list.
        """

        original_sents, processed_sents = [], []
        doc_ids = tools.get_doc_ids(cid, remove_illegal=rm_dialog)  # if rm dialog, rm illegal docs.
        for did in doc_ids:
            doc_fp = tools.get_doc_fp(did)

            # 2d if para_org==True; 1d otherwise.
            original_doc_sents, processed_doc_sents = dataset_parser.doc2sents(fp=doc_fp,
                                                                            #    para_org=config_meta['para_org'],
                                                                               rm_dialog=rm_dialog,
                                                                               rm_stop=rm_stop,
                                                                               stem=stem)

            if max_ns_doc:
                original_doc_sents = original_doc_sents[:max_ns_doc]
                processed_doc_sents = processed_doc_sents[:max_ns_doc]

            original_sents.append(original_doc_sents)
            processed_sents.append(processed_doc_sents)

        return original_sents, processed_sents

    def cid2sents_tdqfs(self, cid):
        cc_dp = join(path_parser.data_tdqfs_sentences, cid)
        fns = [fn for fn in listdir(cc_dp)]
        original_sents, processed_sents = [], []
        for fn in fns:
            sentences = [ss.strip('\n') for ss in io.open(join(cc_dp, fn)).readlines()]
            original_doc_sents, processed_doc_sents = [], []
            for ss in sentences:
                ss_origin = self._proc_sent(ss, rm_dialog=False, rm_stop=False, stem=False)
                ss_proc = self._proc_sent(ss, rm_dialog=False, rm_stop=True, stem=True)

                if ss_proc:
                    original_doc_sents.append(ss_origin)
                    processed_doc_sents.append(ss_proc)
            
            original_sents.append(original_doc_sents)
            processed_sents.append(processed_doc_sents)

        return original_sents, processed_sents

    def cid2paras(self, cid, rm_dialog=True, rm_stop=True, stem=True, max_np_doc=None):
        original_paras, processed_paras = [], []
        doc_ids = tools.get_doc_ids(cid, remove_illegal=rm_dialog)  # if rm dialog, rm illegal docs.
        for did in doc_ids:
            doc_fp = tools.get_doc_fp(did)
            original_doc_paras, processed_doc_paras = dataset_parser.doc2paras(fp=doc_fp,
                                                                               rm_dialog=rm_dialog,
                                                                               rm_stop=rm_stop,
                                                                               stem=stem)

            if max_np_doc:
                original_doc_paras = original_doc_paras[:max_np_doc]
                processed_doc_paras = processed_doc_paras[:max_np_doc]

            original_paras.append(original_doc_paras)
            processed_paras.append(processed_doc_paras)

        return original_paras, processed_paras

    def parse(self, para, clip_and_mask, offset, rm_dialog, rm_stop, stem):
        """
            parse a para and organize results by words.
        """
        sents = [self.word_tokenize(self._proc_sent(sent, rm_dialog, rm_stop, stem))
                 for sent in self.sent_tokenize(para)]

        if not clip_and_mask:  # you only index after clipped
            return sents

        return clip_and_mask(sents, offset, join_query_para=config.join_query_para)

    def parse_query(self, query):
        """
            parse a query string => a dict with keys: ('words', 'sent_mask').
        """
        if 'max_nw_query' not in config_model:
            raise ValueError('Specify max_nw_query in config to clip query!')

        return self.word_tokenize(query)[:config_model['max_nw_query']]
        # return self.parse(para=query, clip_and_mask=cm.clip_and_mask_query_sents,
        #                   offset=1, rm_dialog=False, rm_stop=False, stem=False)

    def parse_doc2paras(self, fp, concat_paras, offset):
        """
            from file, parse a doc => a dict with keys: ('paras', 'doc_masks').

            the value of 'paras': a list of dicts with keys: ('words', 'sent_mask').
        """
        paras = self.get_doc(fp, concat_paras)

        if concat_paras:  # the whole text
            return paras

        # logger.info('fp: {}'.format(fp))
        doc_res = cm.clip_and_mask_a_doc(paras)

        para_list = list()

        for para in doc_res['paras']:
            res = self.parse(para, clip_and_mask=cm.clip_and_mask_para_sents, offset=offset)
            para_list.append(res)

        res = {
            'paras': para_list,
            'doc_masks': doc_res['doc_masks'],
        }

        return res

    def parse_doc2sents(self, fp):
        """

            From file, parse a doc => a dict with keys:
                {'sents', 'doc_masks'}.

            The value of 'sents':
                2D nested list; each list consists of (clipped) sentence words.

            (Initially for QueryNetSL; now also for BertDetectS)
        """
        _, processed_sents = self.doc2sents(fp, para_org=False, rm_dialog=True, rm_stop=False, stem=False)

        if not processed_sents:
            # raise ValueError('No processed sents in fp: {}'.format(fp))
            return None

        sents = [self.sent2words(sent) for sent in processed_sents]

        # origin_ns = len(sents)
        res = cm_sl.clip_and_mask_doc_sents(sents=sents)
        # clip_ns = len(res['words'])
        # logger.info('[clip] #sents: {0} => {1}, {2}'.format(origin_ns, clip_ns, fp))
        return res

    def list_illegal_doc_ids(self):
        """
            list illegal doc ids.

            results: ['2007_D0709B_APW19990124.0079', '2007_D0736H_APW19990311.0174']
        :return:

        """
        years = ['2007']
        for year in years:
            illegal_doc_ids = []

            cc_ids = tools.get_cc_ids(year, model_mode='test')
            for cid in tqdm(cc_ids):
                doc_ids = tools.get_doc_ids(cid, remove_illegal=False)
                for did in doc_ids:
                    if not self.parse_doc2sents(fp=tools.get_doc_fp(did)):
                        illegal_doc_ids.append(did)

            logger.info('[list_illegal_doc_ids] {}: {}'.format(year, illegal_doc_ids))

    def parse_trigger2sents(self, trigger):
        """
            For QueryNetSL (trigger org via sents)
        :param trigger:
        :return:
        """
        sents = [self.sent2words(sent) for sent in trigger]
        res = cm_sl.clip_and_mask_trigger_sents(sents)

        return res

    def parse_trigger2words(self, trigger):
        """
            For QueryNetSL.
        :param trigger:
        :return:
        """
        trigger = self.sent2words(trigger)
        return cm_sl.clip_and_mask_trigger(trigger)

    def parse_lexrank_sent(self, sent):
        return cm.clip_lexrank_sent_words(self.sent2words(sent))

    def parse_paraphrase_sent(self, sent):
        return cm.clip_paraphrase_sent_words(self.sent2words(sent))

    def sent2words(self, sent):
        """
            tokenize the given proprocessed sent.

        :param sent:
        :return:
        """
        # logger.info('sent: {}'.format(sent))
        return self.word_tokenize(sent)

    def trigger_sent2words(self, query_sent):
        return self.word_tokenize(query_sent)

    def parse_summary(self, fp):
        # logger.info('processing fp: {}'.format(fp))
        sent_as_line = fp.split('/')[-2] != '2007'
        # logger.info('sent_as_line: {}'.format(sent_as_line))

        with io.open(fp, encoding='latin1') as f:
            content = f.readlines()

        lines = [ll.rstrip('\n') for ll in content]

        if sent_as_line:
            return lines

        sents = list(itertools.chain(*[self.sent_tokenize(ll) for ll in lines]))
        return sents

    def _get_query_title(self, dataset, query_id):
        return query_info[dataset][query_id]['title']

    @staticmethod
    def _get_query_narr(dataset, query_id):
        return query_info[dataset][query_id]['narr']

    def build_query_info(self, year, tokenize_narr, concat_title_narr=False, proc=True):
        fp = join(path_parser.data_topics, '{}.sgml'.format(year))
        with io.open(fp, encoding='utf-8') as f:
            article = f.read()
        segs = article.split('\n\n\n')
        query_info = dict()
        for seg in segs:
            seg = seg.rstrip('\n')
            if not seg:
                continue
            query_id = re.search(self.id_pat, seg)
            title = re.search(self.title_pat, seg)
            narr = re.search(self.narr_pat, seg)

            if not query_id:
                logger.info('no query id in {0} in {1}...'.format(seg, year))
                break

            if not title:
                raise ValueError('no title in {0}...'.format(seg))
            if not narr:
                raise ValueError('no narr in {0}...'.format(seg))

            query_id = query_id.group()
            title = title.group()
            narr = narr.group()  # containing multiple sentences

            if proc:
                title = self._proc_sent(sent=title, rm_dialog=False, rm_stop=False, stem=True)

            if not title:
                raise ValueError('no title in {0}...'.format(seg))

            if tokenize_narr:
                narr = sent_tokenize(narr)
                if type(narr) != list:
                    narr = [narr]

                if proc:
                    narr = [self._proc_sent(sent=narr_sent, **self.proc_params_for_questions)
                            for narr_sent in narr]
            elif proc:
                    narr = self._proc_sent(sent=narr, **self.proc_params_for_questions)

            if not narr:
                raise ValueError('no narr in {0}...'.format(seg))

            cid = config.SEP.join((year, query_id))

            if not concat_title_narr:
                query_info[cid] = {config.TITLE: title,
                                   config.NARR: narr,  # str or list
                                   }
                continue

            # concat title and narr
            if tokenize_narr:  # list
                narr.insert(0, title)  # narr is a list
                query_info[cid] = narr
            else:  # str
                sep = '. '
                if title.endswith('.'):
                    sep = sep[-1]
                title = 'describe ' + title
                query_info[cid] = sep.join((title, narr))

        return query_info

    def build_trigger_info(self, year, tokenize_narr):
        """
            {
                cid_0: {
                    'query',
                    'headline': {
                        did_0: headline_0,
                        did_1: headline_1,
                },
                cid_1: {
                    ...
                }
            }
        :param year:
        :return:
        """
        trigger_info = {}
        query_info = self.build_query_info(year=year, tokenize_narr=tokenize_narr, concat_title_narr=True)
        headline_info = self.build_headline_info(year=year, tokenize=None, silent=True)

        for cid in query_info:
            trigger_info[cid] = {}

            trigger_info[cid]['query'] = query_info[cid]
            trigger_info[cid]['headline'] = {}

            for did in headline_info:
                if did.startswith(cid):
                    headline = headline_info[did]['headline']
                    trigger_info[cid]['headline'][did] = headline
        return trigger_info

    def build_question_info(self, year, tokenize_question):
        question_info = {}
        query_info = self.build_query_info(year=year, tokenize_narr=tokenize_question, concat_title_narr=True)
        headline_question_info = self.build_headline_question_info(year=year, tokenize_question=tokenize_question)

        for cid in query_info:
            question_info[cid] = {}

            question_info[cid]['query'] = query_info[cid]
            question_info[cid]['headline'] = {}

            for did in headline_question_info:
                if tools.get_cid(did) == cid:
                    headline_question = headline_question_info[did]['headline_question']
                    question_info[cid]['headline'][did] = headline_question
        return question_info

    def get_cid2trigger(self, tokenize_narr):
        trigger_dict = dict()
        for year in config.years:
            annual_dict = self.build_trigger_info(year, tokenize_narr)
            trigger_dict = {
                **annual_dict,
                **trigger_dict,
            }
        return trigger_dict

    def get_cid2query(self, tokenize_narr):
        query_dict = dict()
        for year in config.years:
            annual_dict = self.build_query_info(year, tokenize_narr, concat_title_narr=True)
            query_dict = {
                **annual_dict,
                **query_dict,
            }
        return query_dict

    def get_cid2title(self):
        title_dict = dict()
        for year in config.years:
            annual_dict = self.build_query_info(year, tokenize_narr=False, concat_title_narr=False)
            for cid in annual_dict:
                annual_dict[cid] = annual_dict[cid][config.TITLE]

            title_dict = {
                **annual_dict,
                **title_dict,
            }
        return title_dict

    def get_cid2narr(self):
        title_dict = dict()
        for year in config.years:
            annual_dict = self.build_query_info(year, tokenize_narr=False, concat_title_narr=False)
            for cid in annual_dict:
                annual_dict[cid] = annual_dict[cid][config.NARR]

            title_dict = {
                **annual_dict,
                **title_dict,
            }
        return title_dict

    def get_trigger_list(self, tokenize_narr):
        triggers = []
        for year in config.years:
            trigger_info = self.build_trigger_info(year, tokenize_narr)
            for cid in trigger_info:
                triggers.append(trigger_info[cid]['query'])

                headlines = trigger_info[cid]['headline'].values()
                if tokenize_narr:  # each trigger is a list
                    headlines = [[hh] for hh in headlines]
                triggers.extend(headlines)

        return triggers

    def build_query_dict(self, year, query_type):
        if query_type == 'narr':
            return self.build_query_info(year)
        elif query_type == 'headline':
            return self.build_headline_info(year)
        else:
            raise ValueError('Unknown query type {0}'.format(query_type))

    def _build_cluster_info(self, dataset):
        """
            return nested dict:
            cluster_info = {
                dataset: {
                    query: {
                        "title": TITLE,
                        "narr": NARR,
                    }
                }
            }
        """
        fp = join(path_parser.data_topics, '{0}_topics.sgml'.format(dataset))
        with open(fp, encoding='utf-8') as f:
            text = f.read()
        segs = text.split('\n\n\n')
        cluster_info = dict()
        for seg in segs:
            seg = seg.rstrip('\n')
            if not seg:
                continue
            query_id = re.search(self.id_pat, seg)
            title = re.search(self.title_pat, seg)
            narr = re.search(self.narr_pat, seg)

            if not query_id:
                logger.warning('no query id in {0} in {1}...'.format(seg, dataset))
                break

            if not title:
                logger.warning('no title in {0}...'.format(seg))
                break
            if not narr:
                logger.warning('no narr in {0}...'.format(seg))
                break

            query_id = query_id.group()
            title = title.group()
            narr = narr.group()
            narr = self._proc_sent(narr)

            cluster_info[query_id] = {
                'title': title,
                'narr': narr,
            }

        return cluster_info

    def _build_article_info(self, dataset):
        """
            return nested dict:
            article_info = {
                dataset: {
                    q_id: {
                        fn: {
                            "headline": HEADLINE,
                            "text": TEXT,
                        }
                    }
                }
            }
        """
        article_info = dict()
        root = join(path_parser.data_docs, '{}_docs'.format(dataset))
        queries = [query_id for query_id in listdir(root) if isdir(join(root, query_id))]
        # print 'queries: {}'.format(queries)
        for q_id in queries:
            article_info_q = dict()
            query_dp = join(root, q_id)
            doc_fns = [doc_fn for doc_fn in listdir(query_dp) if isfile(join(query_dp, doc_fn))]
            for fn in doc_fns:
                fp = join(query_dp, fn)
                headline = self._get_headline(dataset, fp)
                text = self.get_article(fp)
                # print '----------------------------------------------'
                # print 'headline: {0}\nfp: {1}'.format(headline, fp)
                article_info_q[fn] = {
                    'headline': headline,
                    'text': text,
                }
            article_info[q_id] = article_info_q

        return article_info

    def build_all_info(self):
        for year in config.years:
            self.cluster_info[year] = self._build_cluster_info(year)
            self.article_info[year] = self._build_article_info(year)


dataset_parser = DatasetParser()

if __name__ == '__main__':
    dataset_parser = DatasetParser()
    question_info = dataset_parser.build_question_info(year='2007', tokenize_question=True)
    print(question_info)
