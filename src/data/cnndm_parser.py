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


import os
import hashlib
import struct
import subprocess
import collections
"""

    This file provides a CnnDmParser class for getting doc sentences and summary sentences from CNN/DM raw data.


"""

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence
# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

"""
    Following functions are borrowed from https://github.com/becxer/cnn-dailymail/blob/master/make_datafiles.py.

"""

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def read_text_file(text_file):
    
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def get_art_abs(story_file):
    """
        Revised to get doc and summary sentences.
        
    """

    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)
    summary = ' '.join(highlights)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    # abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])
    # return article, abstract
    return article, summary


class CnnDmParser():
    def __init__(self, dataset_var):
        url_list = read_text_file(path_parser.cnndm_url / f'all_{dataset_var}.txt')
        url_hashes = get_url_hashes(url_list)
        self.story_fnames = [s+".story" for s in url_hashes]
        self.num_stories = len(self.story_fnames)
    
    def get_sentences(self, story):
        return nltk.tokenize.sent_tokenize(story)

    def sample_generator(self):
        """
            Generate a sample
        """
        for idx, s in enumerate(self.story_fnames):
            # if idx % 1000 == 0:
            # print("Writing story %i of %i; %.2f percent done" % (idx, self.num_stories, float(idx)*100.0/float(self.num_stories)))
            if os.path.isfile(os.path.join(path_parser.cnndm_raw_cnn_story, s)):
                story_file = os.path.join(path_parser.cnndm_raw_cnn_story, s)
            elif os.path.isfile(os.path.join(path_parser.cnndm_raw_dm_story, s)):
                story_file = os.path.join(path_parser.cnndm_raw_dm_story, s)
            else:
                raise Exception(f'Cannot find story: {s}')

            # Get the strings to write to .bin file
            article, summary = get_art_abs(story_file)
            sentences = self.get_sentences(article)
            yield (idx, sentences, summary)


if __name__ == "__main__":
    parser = CnnDmParser(dataset_var='train')
    sample_generator = parser.sample_generator()
    idx, sentences, summary = next(sample_generator)
    logger.info(f'sentences: {sentences}\nsummary: {summary}\n\n')