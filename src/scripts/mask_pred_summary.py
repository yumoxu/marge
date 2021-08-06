# -*- coding: utf-8 -*-
import os
import io
from os.path import dirname, abspath, join, exists
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

from multiprocessing import Pool
import scripts.query_stats as query_stats
import random
from copy import copy
import numpy as np

import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
from scripts.mask_summary_with_ratio import SummaryObject

"""
    This module is for masking system-produced summaries, e.g., for query expansion purpose.

"""

if config.path_type == 'afs':
    DP_PROJ = Path('/disk/nfs/ostrom/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_DATA = DP_PROJ / 'data'

# ir-dial-tf-tdqfs-0.6_cos-nw_250
# grsum-tdqfs-0.6_cos-0_wan-nw_250
# grsum-tdqfs-1.0_cos-0_wan-nw_250
# centrality-hard_bias-0.85_damp-ir-tf-2006-0.6_cos-10_wan
# centrality-hard_bias-0.85_damp-ir-tf-2007-0.6_cos-10_wan
# rr-39_config-26000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-0.6_cos-nw_250
# rr-39_config-26000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-0.6_cos-nw_250
# rr-34_config-25000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-0.6_cos-nw_250
dataset = 'grsum-tdqfs-0.6_cos-0_wan-nw_250'
DP_SRC = DP_PROJ / 'text' / dataset  # to build raw
if not os.path.exists(DP_SRC):
    raise ValueError(f'Build DP_SRC first: {DP_SRC}')

MAX_KEY = 'ns'  # ns or nw
MAX_VAL = 20
if MAX_KEY == 'ns' or MAX_VAL != 250:
    items = dataset.split('-')[:-1]
    items.append(f'{MAX_KEY}_{MAX_VAL}')
    dataset = '-'.join(items)

DP_PRED = DP_PROJ / 'pred' / dataset  # root for raw, parsed, and masked
if not os.path.exists(DP_PRED):
    os.mkdir(DP_PRED)

FP_RAW_PRED = DP_PRED / 'raw.json'
FP_PARSED_PRED_SUMMARY = DP_PRED / 'parsed_summary.json'


def build_raw(dataset):
    records = []
    for fn in os.listdir(DP_SRC):
        lines = io.open(DP_SRC/fn).readlines()
        count, text = 0, ''

        if MAX_KEY == 'nw':
            for idx, line in enumerate(lines):
                count += len(word_tokenize(line))
                line = line.strip('\n')
                if line[-1].isalpha():
                    line += '.'
                text += line
                text += ' '
                if count >= MAX_VAL:
                    records.append({'cid': fn, 'raw': text[:-1]})  # remove the last space
                    break
        
        elif MAX_KEY == 'ns':
            for line in lines[:MAX_VAL]:
                line = line.strip('\n')
                if line[-1].isalpha():
                    line += '.'
                text += line 
                text += ' '
            records.append({'cid': fn, 'raw': text})
            # if idx >= MAX_VAL:
            #     records.append({'cid': fn, 'raw': text[:-1]})  # remove the last space
            #     break
            # text += line.strip('\n') + ' '
    
    with open(FP_RAW_PRED, 'a') as f:
        for rec in tqdm(records):
            f.write(json.dumps(rec, ensure_ascii=False)+'\n')


class SummaryParser:
    def __init__(self):
        self.raw_fp = FP_RAW_PRED
        self.parse_fp = FP_PARSED_PRED_SUMMARY

        self.raw_summaries = []
        for line in open(self.raw_fp).readlines():
            json_obj = json.loads(line.rstrip('\n'))
            self.raw_summaries.append((json_obj['cid'], json_obj['raw']))

        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
    
    def _parse_sentence(self, sentence):
        result = self.predictor.predict(sentence=sentence)
        result['raw_sentence'] = sentence
        return result

    def _parse(self, cid, raw_summary):
        summary_sentences = sent_tokenize(raw_summary)
        open_ie = []
        for sentence in summary_sentences:
            result = self.predictor.predict(sentence=sentence)
            result['raw_sentence'] = sentence
            open_ie.append(result)

        output = {
            'cid': cid,
            'raw_summary': raw_summary,
            'open_ie': open_ie,
        }

        return output

    def parse(self):
        parse_dicts = []
        for cid, raw_summary in self.raw_summaries:
            json.loads()
            output = self._parse(cid, raw_summary)
            parse_dicts.append(output)
        
        return parse_dicts

    def dump_parse(self):
        with open(self.parse_fp, 'a') as f:
            for pd in self.parse_dicts:
                f.write(json.dumps(pd, ensure_ascii=False)+'\n')

    def parse_and_dump(self):
        parse_dicts = []

        with open(self.parse_fp, 'a') as f:
            for cid, raw_summary in tqdm(self.raw_summaries):
                output = self._parse(cid, raw_summary=raw_summary)
                f.write(json.dumps(output, ensure_ascii=False)+'\n')
        
        return parse_dicts


class SummaryMasker:
    def __init__(self, rand_expose, 
            verb_as_slot, 
            min_nw_mode: str, 
            reveal_ratio=1.0,
            max_rand_trials=100, 
            mask_token='[MASK]',
            remove_base=None,
            func_reveal_ratio=1.0):
        self.rand_expose = rand_expose
        self.verb_as_slot = verb_as_slot
        self.max_rand_trials = max_rand_trials
        self.mask_token = mask_token
        self.min_nw_mode = min_nw_mode
        self.reveal_ratio = reveal_ratio
        self.remove_base = remove_base
        self.func_reveal_ratio = func_reveal_ratio

        self.parse_fp = FP_PARSED_PRED_SUMMARY
        mask_fn = f'masked-{min_nw_mode}-reveal_{self.reveal_ratio}'
        if self.verb_as_slot:
            mask_fn += '-verb_as_slot'
        
        if self.remove_base:
            mask_fn += '-remove_base'

        if not self.rand_expose:
            mask_fn += '-no_rand_expose'
        
        if func_reveal_ratio < 1.0:
            mask_fn += f'-func_reveal_{func_reveal_ratio}'

        self.mask_fp = join(DP_PRED, f'{mask_fn}.json')
        self.masked_dicts = self.mask()
    
    def get_summary_obj(self, line):
        summary_obj = SummaryObject(line=line, 
                                    rand_expose=self.rand_expose, 
                                    verb_as_slot=self.verb_as_slot, 
                                    min_nw_mode=self.min_nw_mode, 
                                    reveal_ratio=self.reveal_ratio,
                                    max_rand_trials=self.max_rand_trials,
                                    remove_base=self.remove_base,
                                    func_reveal_ratio=self.func_reveal_ratio)
        return summary_obj

    def get_record(self, cid, summary_obj):
        """
            Todo: save masked words, original words/sentences/summary. 
        """
        s_dicts = []
        for so in summary_obj.sentence_objects:
            s_dict = {
                'raw_sentence': so.raw_sentence,
                'words': so.words,
                'masked_words': so.masked_words,
                'merged_masked_words': so.merged_masked_words,
                'n_revealed': so.n_revealed,
            }
            s_dicts.append(s_dict)

        global_n_revealed = summary_obj.get_global_n_revealed()
        record = {
            'cid': cid,
            'original_summary': summary_obj.raw_summary,
            'masked_seq': summary_obj.masked_seq,
            'masked_seq_with_sep': summary_obj.masked_seq_with_sep,
            'masked_seq_with_sub': summary_obj.masked_seq_with_sub,
            'global_n_revealed': global_n_revealed,
            'sentences': s_dicts,
        }
        return record

    def mask(self):
        masked_dicts = []
        n_sentences_wo_init_non_abs = 0.0
        n_sentences = 0.0
        with open(self.parse_fp) as f:
            for line in f:
                line = line.rstrip('\n')
                cid = json.loads(line)['cid']
                summary_obj = self.get_summary_obj(line=line)

                n_sentences_wo_init_non_abs += len([so for so in summary_obj.sentence_objects if so.no_init_abs_non_span])
                n_sentences += len(summary_obj.sentence_objects)

                summary_obj.iteratively_reveal()
                masked_dict = self.get_record(cid, summary_obj)
                masked_dicts.append(masked_dict)
        
        self.n_sentences_wo_init_non_abs = n_sentences_wo_init_non_abs
        self.n_sentences = n_sentences
        return masked_dicts
        
    def dump_mask(self):
        with open(self.mask_fp, 'a') as f:
            for md in self.masked_dicts:
                f.write(json.dumps(md, ensure_ascii=False)+'\n')
        print(f'{len(self.masked_dicts)} masked records have been dumped to {self.mask_fp}')


def parse():
    s_parser = SummaryParser()
    s_parser.parse_and_dump()


def mask(rand_expose, verb_as_slot, min_nw_mode, reveal_ratio, remove_base, func_reveal_ratio):
    s_masker = SummaryMasker(rand_expose=rand_expose, 
        verb_as_slot=verb_as_slot, 
        min_nw_mode=min_nw_mode,
        reveal_ratio=reveal_ratio,
        max_rand_trials=100,
        remove_base=remove_base,
        func_reveal_ratio=func_reveal_ratio)
    s_masker.dump_mask()



if __name__ == "__main__":
    # build_raw(dataset=dataset)
    # parse()

    params = {
        'rand_expose': True,
        'verb_as_slot': False,
        'min_nw_mode': 'ratio',  # max, sample
        'reveal_ratio': 1.0,
        'remove_base': False,  # False
        'func_reveal_ratio': 1.0,
    }
    mask(**params)
    