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
class MultiNewsParser():
    def __init__(self, dataset_var):
        # self.STORY_SEP = 'story_separator_special_tag'
        # self.SENTENCE_SEP = ' ' * 5
        self.STORY_SEP = ' ||||| '
        src_fp = path_parser.multinews_raw / f'{dataset_var}.src.cleaned'
        src_lines = open(src_fp).readlines()

        tgt_fp = path_parser.multinews_raw / f'{dataset_var}.tgt'
        tgt_lines = open(tgt_fp).readlines()

        cluster_size = len(src_lines)
        assert cluster_size == len(tgt_lines)

        self.src_tgt_tuples = zip(list(range(cluster_size)), src_lines, tgt_lines)
        
    def get_stories(self, line):
        return line.split(self.STORY_SEP)
    
    def get_sentences(self, story):
        return nltk.tokenize.sent_tokenize(story)

    def get_summary(self, line):
        return line[2:]

    def sample_generator(self):
        """
            Generate a sample:
                cluster_idx, sentences (2D, organized via stories), summary
        """
        for cluster_idx, src, tgt in self.src_tgt_tuples:
            stories = self.get_stories(line=src)
            sentences = [self.get_sentences(story=st) for st in stories]
            summary = self.get_summary(line=tgt)
            yield (cluster_idx, sentences, summary)
