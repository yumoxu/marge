# -*- coding: utf-8 -*-
import os
from os.path import dirname, abspath, join
from pathlib import Path
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

import re
from os import listdir
import itertools
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize

from allennlp.predictors.predictor import Predictor

DP_PROJ = Path('~/margesum')
DP_DATA = DP_PROJ / 'data'
DP_RAW_QUERY = DP_DATA / 'raw_query'
DP_PARSED_QUERY = DP_DATA / 'parsed_query'
DP_MASKED_QUERY = DP_DATA / 'masked_query'

YEARS = ['2005', '2006', '2007']
SEP = '_'
TITLE = 'title'
NARR = 'narr'


class QueryInfo:
    def __init__(self, year):
        self.year = year
        BASE_PAT = '(?<=<{0}> )[\s\S]*?(?= </{0}>)'
        BASE_PAT_WITH_NEW_LINE = '(?<=<{0}>\n)[\s\S]*?(?=\n</{0}>)'
        self.ID_PAT = re.compile(BASE_PAT.format('num'))
        self.TITLE_PAT = re.compile(BASE_PAT.format('title'))
        self.NARR_PAT = re.compile(BASE_PAT_WITH_NEW_LINE.format('narr'))

        self.query_fp = join(DP_RAW_QUERY, '{}.sgml'.format(year))
        self.cid2query_sentences = self.build_cid2query_sentences()

    def build_cid2query_sentences(self):
        with open(self.query_fp) as f:
            article = f.read()
        segs = article.split('\n\n\n')
        query_info = dict()
        for seg in segs:
            seg = seg.rstrip('\n')
            if not seg:
                continue
            query_id = re.search(self.ID_PAT, seg)
            title = re.search(self.TITLE_PAT, seg)
            narr = re.search(self.NARR_PAT, seg)

            if not query_id:
                print('no query id in {0} in {1}...'.format(seg, year))
                break

            if not title:
                raise ValueError('no title in {0}...'.format(seg))
            if not narr:
                raise ValueError('no narr in {0}...'.format(seg))

            query_id = query_id.group()
            cid = SEP.join((self.year, query_id))

            title = title.group()
            title = f'Describe {title}'

            narr = narr.group()  # containing multiple sentences
            narr = narr.replace('\n', ' ')
            narr = sent_tokenize(narr)
            if type(narr) != list:
                narr = [narr]
            narr.insert(0, title)
            
            query_info[cid] = narr

        return query_info


class QueryParser:
    def __init__(self, year):
        self.parse_fp = join(DP_PARSED_QUERY, '{}.json'.format(year))
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
        self.cid2query_sentences = QueryInfo(year).cid2query_sentences
        self.parse_dicts = self.parse()
        
    def parse(self):
        outputs = []

        for cid, query_sentences in tqdm(self.cid2query_sentences.items()):
            output = {
                'cid': cid,
                'raw_query': query_sentences,
            }
            open_ie = []
            for sentence in query_sentences:
                # {
                #     "tokens": [...],
                #     "tag_spans": [{"ARG0": "...", "V": "...", "ARG1": "...", ...}]
                # }
                result = self.predictor.predict(sentence=sentence)
                # print(f'result: {result}')
                result['sentence'] = sentence
                open_ie.append(result)
            output['open_ie'] = open_ie
            outputs.append(output)

        return outputs
    
    def dump_parse(self):
        with open(self.parse_fp, 'a') as f:
            for pd in self.parse_dicts:
                f.write(json.dumps(pd, ensure_ascii=False)+'\n')


class QueryMasker:
    def __init__(self, year, sub_token='[SUBQUERY]'):
        # self.parse_fp = join(DP_PARSED_QUERY, '{}.json'.format(year))
        self.command_words = ['describe', 'identify', 'explain', 'name', 'report', 'include', 'provide', 'discuss', 'note', 'specify', 'give', 'determine', 'summarize', 'track', 'relate', 'write']
        
        # self.query_words_anywhere = list(itertools.chain(*[self.command_words, self.question_words]))
        self.question_words_wh = ['what', 'where', 'why', 'when', 'who', 'which', 'how', 'whom']
        self.query_words_binary = ['is', 'are', 'have', 'do']  # binary questions

        self.mask_token = '[MASK]'
        self.sub_token = sub_token

        self.cid2query_sentences = QueryInfo(year).cid2query_sentences
        self.mask_fp = join(DP_MASKED_QUERY, '{}.json'.format(year))
        self.masked_dicts = self.mask()

    def _mask_question(self, sentence):
        has_question_words = False
        for qww in self.question_words_wh:
            if qww in sentence:
                sentence = sentence.replace(qww, self.mask_token)
                has_question_words = True

        for qwb in self.query_words_binary:
            if sentence.startswith(qwb):
                sentence = sentence.replace(qwb, self.mask_token)
                has_question_words = True

        return sentence, has_question_words

    def _mask_command(self, sentence):
        for cw in self.command_words:
            cw += ' '  # make sure it is a commanding word. e.g., not "related to..."
            if cw in sentence:
                sentence = sentence.replace(cw, self.mask_token+' ')
        
        return sentence

    def _mask(self, sentence):
        """
            for each sentence, check if it is a question-type query.
            
            If it is not, mask the command words in it.
        """
        sentence = sentence.replace('?', '.').lower()
        sentence, has_question_words = self._mask_question(sentence)
        if has_question_words:
            return sentence
        
        return self._mask_command(sentence)

    def mask_with_sub(self, masked_sentences):
        with_sub = ''
        for idx, ms in enumerate(masked_sentences):
            if ms.endswith('.'):
                ms = ms[:-1]
            with_sub += ms

            if idx < len(masked_sentences)-1:  # the last token should be appended at after being trimmed (potentially)
                with_sub += ' ' + self.sub_token + ' '
        
        return with_sub
        
    def mask(self):
        outputs = []

        for cid, query_sentences in tqdm(self.cid2query_sentences.items()):
            masked_sentences = [self._mask(sentence=sentence) for sentence in query_sentences]
            with_sub = self.mask_with_sub(masked_sentences)
            output = {
                'cid': cid,
                'raw_query': query_sentences,
                'masked_query': masked_sentences,
                'masked_query_with_sub': with_sub,
            }
            outputs.append(output)

        return outputs 

    def dump_mask(self):
        with open(self.mask_fp, 'a') as f:
            for md in self.masked_dicts:
                f.write(json.dumps(md, ensure_ascii=False)+'\n')


def mask_and_dump_e2e():
    for year in YEARS:
        q_masker = QueryMasker(year=year)
        q_masker.dump_mask()


if __name__ == "__main__":
    mask_and_dump_e2e()
