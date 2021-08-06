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
from data.cnndm_parser import CnnDmParser
from functools import partial


"""
    This file is for ablation study -InfoExt. 
    Two random modes are provided:
        - randomly mask a ratio of all tokens, e.g., 15%
        - randomly mask #functional words, e.g., consistent to our best masking mechansim 

"""
random.seed(2020)
np.random.seed(2020)

if config.path_type == 'afs':
    DP_PROJ = Path('/disk/scratch/margesum')
else:
    DP_PROJ = Path('~/margesum')

DP_DATA = DP_PROJ / 'data'
DP_RAW_MN = DP_DATA / 'multinews' / 'raw'
DP_PARSED_MN_SUMMARY = DP_DATA / 'multinews' / 'parsed_mn_summary'
DP_MASKED_MN_SUMMARY = DP_DATA / 'multinews'/ 'masked_mn_summary'

DP_PARSED_CNNDM_SUMMARY = DP_DATA / 'cnndm' / 'parsed_cnndm_summary'
DP_MASKED_CNNDM_SUMMARY = DP_DATA / 'cnndm' / 'masked_cnndm_summary'

OPENIE_PATH = 'https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz'
DATASET_VARS = ['train', 'val', 'test', 'debug']

MN_DATASET_SIZE = {
    'train': 44972,
    'val': 5622,
    'test': 5622,
    'debug': 5000,
}

CNNDM_DATASET_SIZE = {
    'train': 287227,
    'val': 13368,
    'test': 11490,
    'debug': 5000,
}

class SummaryParser:
    def __init__(self, dataset_var):
        self.raw_fp = join(DP_RAW_MN, f'{dataset_var}.tgt')
        self.parse_fp = join(DP_PARSED_MN_SUMMARY, f'{dataset_var}.json')

        self.raw_summaries = open(self.raw_fp).readlines()
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
    
    def _parse_sentence(self, sentence):
        result = self.predictor.predict(sentence=sentence)
        result['raw_sentence'] = sentence
        return result

    def _parse(self, raw_summary):
        raw_summary = raw_summary[2:]  # remove "- " at beginning
        summary_sentences = sent_tokenize(raw_summary)
        open_ie = []
        for sentence in summary_sentences:
            result = self.predictor.predict(sentence=sentence)
            result['raw_sentence'] = sentence
            open_ie.append(result)

        output = {
            'raw_summary': raw_summary,
            'open_ie': open_ie,
        }

        return output

    def parse(self):
        parse_dicts = []

        for raw_summary in tqdm(self.raw_summaries):
            output = self._parse(raw_summary=raw_summary)
            parse_dicts.append(output)
        
        return parse_dicts

    def dump_parse(self):
        with open(self.parse_fp, 'a') as f:
            for pd in self.parse_dicts:
                f.write(json.dumps(pd, ensure_ascii=False)+'\n')

    def parse_and_dump(self):
        parse_dicts = []

        with open(self.parse_fp, 'a') as f:
            for raw_summary in tqdm(self.raw_summaries):
                output = self._parse(raw_summary=raw_summary)
                f.write(json.dumps(output, ensure_ascii=False)+'\n')
        
        return parse_dicts


class CnnDmSummaryParser:
    def __init__(self, dataset_var):
        self.sample_gen = CnnDmParser(dataset_var=dataset_var).sample_generator()
        self.n_summary = CNNDM_DATASET_SIZE[dataset_var]
        self.predictor = Predictor.from_path(OPENIE_PATH)
    
    def _parse_sentence(self, sentence):
        result = self.predictor.predict(sentence=sentence)
        result['raw_sentence'] = sentence
        return result

    def _parse(self, raw_summary):
        summary_sentences = sent_tokenize(raw_summary)
        open_ie = []
        for sentence in summary_sentences:
            result = self.predictor.predict(sentence=sentence)
            result['raw_sentence'] = sentence
            open_ie.append(result)

        output = {
            'raw_summary': raw_summary,
            'open_ie': open_ie,
        }

        return output

    def parse(self):
        parse_dicts = []

        for raw_summary in tqdm(self.raw_summaries):
            output = self._parse(raw_summary=raw_summary)
            parse_dicts.append(output)
        
        return parse_dicts

    def dump_parse(self):
        with open(self.parse_fp, 'a') as f:
            for pd in self.parse_dicts:
                f.write(json.dumps(pd, ensure_ascii=False)+'\n')

    def parse_and_dump(self, start=0, end=-1):
        if end != -1:
            total = end
            parse_fp = DP_PARSED_CNNDM_SUMMARY / f'{dataset_var}_{start}_{end}.json'
        else:
            total = self.n_summary
            parse_fp = DP_PARSED_CNNDM_SUMMARY / f'{dataset_var}_{start}_EOS.json'

        # self.parse_fp = join(DP_PARSED_CNNDM_SUMMARY, f'{dataset_var}.json')
        with open(parse_fp, 'a') as f:
            for doc_idx, _, summary in tqdm(self.sample_gen, total=total):
                if doc_idx < start:
                    continue

                if end != -1 and doc_idx >= end:
                    break

                output = self._parse(raw_summary=summary)
                f.write(json.dumps(output, ensure_ascii=False)+'\n')


class Slot():
    def __init__(self, slot_id, span):
        self.slot_id = slot_id
        self.span = span
        self.status = 0
        

class Fact:
    def __init__(self, tags: list, verb_as_slot: bool):
        """
            slots: 2D list, slot indices, organized via slots
            
            slot_flat: 1D list, slot indices, flat
            non_slot_flat: 1D list, non-slot indices, flat
            
            slot_status: 1D list, status tracking for each slot. 1 for used while 0 for available.
            
        """
        self.n_words = len(tags)
        self.verb = [idx for idx, tag in enumerate(tags) if tag.endswith('V')]
        self.args = []
        for arg_token in ('ARG0', 'ARG1', 'ARG2'):
            arg = [idx for idx, tag in enumerate(tags) if tag.endswith(arg_token)]
            if arg:
               self.args.append(arg)
        
        if verb_as_slot:
            self.spans = [self.verb] + self.args
        else:
            self.spans = copy(self.args)

        self.slots = [Slot(slot_id=slot_id, span=span) for slot_id, span in enumerate(self.spans)]
        self.span_flat = list(itertools.chain(*self.spans))
        self.non_span_flat = [idx for idx in range(self.n_words) if idx not in self.span_flat]
        # self.slot_status = [0] * len(self.slots)
        
    def get_available_slots(self):
        return [slot for slot in self.slots if slot.status==0]

    def get_available_slot_ids(self):
        return [slot.slot_id for slot in self.get_available_slots()]

    def is_available(self):
        """
            A fact is available when it has at least one available slot.
        """
        available_slots = self.get_available_slots()
        if available_slots:
            return True
        else:
            return False

    def tick_slot(self, slot):
        """
            Tick a slot and its subset slots.
        """
        if slot.status == 1:
            raise ValueError(f'Trying reusing a used slot: {slot.slot_id}')
        
        slot.status = 1
        n_ticks = 1
        
        available_slots = self.get_available_slots()
        for sl in available_slots:
            if set(sl.span).intersection(set(slot.span)) == set(sl.span):  
                sl.status = 1
                n_ticks += 1
        
        return n_ticks


class SentenceObject:
    def __init__(self, open_ie_dict, rand_expose: bool, verb_as_slot: bool, 
            mask_token='[MASK]'):
        """
            open_ie_dict: a dictionary storing OpenIE parsing results
            rand_expose: when init non_fids=None
            verb_as_slot: whether include verbs as slots
        """

        self.raw_sentence = open_ie_dict['raw_sentence']
        self.words = open_ie_dict['words']
        self.n_words = len(self.words)

        self.facts = [Fact(tags=verb_dict['tags'], verb_as_slot=verb_as_slot) for verb_dict in open_ie_dict['verbs']]

        self.rand_expose = rand_expose
        self.mask_token = mask_token
        
        # init masked words and n_revealed; to be updated during iterative revealing.
        self.masked_words, self.n_revealed, self.n_abs_non = self.init_masked_words()
        self.n_masked_func_words = self.n_abs_non - self.n_revealed
        self.merged_masked_words = None

        self.no_init_abs_non_span = False
    
    def get_abs_non_span(self):
        """
            abs_non_span: word indices where no fact exists.

        """
        abs_non_span = set(range(self.n_words))  # init void with all indices

        for fact in self.facts:
            if not abs_non_span:
                break
            abs_non_span = abs_non_span.intersection(fact.non_span_flat)
        
        # no absolute non-slot, expose a random arg as non_slot_flat
        if not abs_non_span:
            self.no_init_abs_non_span = True
            if self.rand_expose:
                _slot, _ = self.rand_slot()
                abs_non_span = _slot.span
        
        return abs_non_span

    def get_available_facts(self):
        available_facts = [fact for fact in self.facts if fact.is_available()]
        return available_facts

    def is_available(self):
        """
            # A sentence is available when it has at least one available fact.
            A sentence is available when it has at least one available mask token.
        """
        if self.n_revealed < self.n_words - self.n_masked_func_words:
            return True
        else:
            return False
    
    def rand_slot(self):
        """
            Randomly select an avalable slot from a random available fact.

            Return None when no fact is available.

        """
        available_facts = self.get_available_facts()
        if not available_facts:
            raise ValueError(f'This sentence has no more available fact. Validate before using this function.  words: {self.words}, masked_words: {self.masked_words}')
        
        _fact = random.choice(available_facts)
        available_slots = _fact.get_available_slots()
        for slot in available_slots:
            assert slot.status == 0

        _slot = random.choice(available_slots)

        n_ticks = _fact.tick_slot(_slot)
        return _slot, n_ticks

    def fill(self, indices, src, target):
        """
            Fill target with src at the given indices.

            n_filled: number of indices newly filled.
        """
        n_filled = 0
        for idx in indices:
            if target[idx] == self.mask_token:
                target[idx] = src[idx]
                n_filled += 1
        return n_filled

    def init_masked_words(self):
        abs_non_span = self.get_abs_non_span()
        n_abs_non = len(abs_non_span)
        n_reveal = len(abs_non_span)

        masked_words = [self.mask_token] * len(self.words)  # init masked words
        return masked_words, n_reveal, n_abs_non

    def merge(self):        
        merged_masked_words = []
        last = None
        for word in self.masked_words:
            if word == self.mask_token and last == self.mask_token:
                last = word
                continue
            last = word
            merged_masked_words.append(word)

        self.merged_masked_words = merged_masked_words


class SummaryObject:
    def __init__(self, line, 
            rand_expose: bool, 
            verb_as_slot: bool, 
            reveal_num_mode: str,
            reveal_ratio: float,
            mask_token='[MASK]', 
            sub_token='[SUBQUERY]',
            max_nw=None):
        """
            Inputs: 
                line: a json string
                rand_expose: when init non_fids=None
                verb_as_slot: whether include verbs as slots
                mask_token: 
                max_nw: 

            Attributes:
                n_sentences: #sentences in summary
                nw: total #words in summary
                max_nw: max #words for final clipping
                merged_masked_words: 2D list; masked words for each sentence (after merging)
                masked_seq: 1D list; a flat version of merged_masked_words.
        """
        self.rand_expose = rand_expose
        self.verb_as_slot = verb_as_slot
        self.reveal_num_mode = reveal_num_mode
        self.reveal_ratio = reveal_ratio

        self.mask_token = mask_token
        
        summary_info = json.loads(line)
        self.raw_summary = summary_info['raw_summary']
        self.sentence_objects = self.get_sentence_objects(open_ie_dicts=summary_info['open_ie'])        
        self.n_sentences = len(self.sentence_objects)

        self.nw = sum([so.n_words for so in self.sentence_objects])
        global_n_abs_non = self.get_global_n_abs_non()
        
        if self.reveal_num_mode == 'n_abs_non':
            self.num_to_reveal = global_n_abs_non
        elif self.reveal_num_mode == 'ratio':
            self.num_to_reveal = int(self.nw * self.reveal_ratio)
        else:
            raise ValueError(f'Invalid reveal_num_mode: {self.reveal_num_mode}')
        print(f'Init: nw: {self.nw}: global_n_abs_non: {global_n_abs_non}, num_to_reveal: {self.num_to_reveal}')

        self.max_nw = max_nw  # for final clipping
        self.merged_masked_words = None
        self.masked_seq = []
        self.sep_token = '[SEP]'
        self.masked_seq_with_sep = []  # with [SEP] between summary sentences

        self.sub_token = sub_token
        self.masked_seq_with_sub = []  # with [SUBQUERY] between summary sentences

    def get_sentence_objects(self, open_ie_dicts):
        sentence_objects = []
        params = {
            'rand_expose': self.rand_expose,
            'verb_as_slot': self.verb_as_slot,
            'mask_token': self.mask_token,
        }

        for open_ie_dict in open_ie_dicts:
            params['open_ie_dict'] = open_ie_dict
            sentence_objects.append(SentenceObject(**params))
        return sentence_objects

    def get_global_n_revealed(self):
        global_n_revealed = sum([so.n_revealed for so in self.sentence_objects])
        return global_n_revealed

    def get_global_n_abs_non(self):
        global_n_abs_non = sum([so.n_abs_non for so in self.sentence_objects])
        return global_n_abs_non

    def finalize(self):
        """
            Merge and clip the whole summary.

        """
        for so in self.sentence_objects:
            so.merge()

        self.merged_masked_words = [so.merged_masked_words for so in self.sentence_objects if so.merged_masked_words]  # filter empty sentences
        
        for sid, words in enumerate(self.merged_masked_words):
            self.masked_seq += words
            self.masked_seq_with_sep += words
            self.masked_seq_with_sub += words
            
            if sid < len(self.merged_masked_words)-1:
                # since more tokens may be inserted and the final length is not set
                # we leave the last [SEP] to later BERT input proc 
                self.masked_seq_with_sep += [self.sep_token]
                self.masked_seq_with_sub += [self.sub_token]
        
        if self.max_nw:
            self.masked_seq = self.masked_seq[:self.max_nw]
            self.masked_seq_with_sep = self.masked_seq_with_sep[:self.max_nw]
            self.masked_seq_with_sub = self.masked_seq_with_sub[:self.max_nw]

    def reveal(self):
        indice_pool= []
        for sid, so in enumerate(self.sentence_objects):
            indice_pool.extend([(sid, wid) for wid in range(so.n_words)])
        
        sampled_indices = random.sample(indice_pool, self.num_to_reveal)
        for sid, so in enumerate(self.sentence_objects):
            indices_to_reveal = [_wid for _sid, _wid in sampled_indices if sid == _sid]
            so.fill(indices=indices_to_reveal, src=so.words, target=so.masked_words)
        self.finalize()


class SummaryMasker:
    def __init__(self, dataset, dataset_var, 
            rand_expose, 
            verb_as_slot, 
            reveal_num_mode,
            reveal_ratio,
            mask_token='[MASK]'):
        self.dataset_var = dataset_var
        self.rand_expose = rand_expose
        self.verb_as_slot = verb_as_slot
        self.reveal_num_mode = reveal_num_mode
        self.reveal_ratio = reveal_ratio
        self.mask_token = mask_token

        if dataset == 'mn':
            dp_parsed = DP_PARSED_MN_SUMMARY
            dp_mask = DP_MASKED_MN_SUMMARY
            self.n_summary = MN_DATASET_SIZE[dataset_var]
        elif dataset == 'cnndm':
            dp_parsed = DP_PARSED_CNNDM_SUMMARY
            dp_mask = DP_MASKED_CNNDM_SUMMARY
            self.n_summary = CNNDM_DATASET_SIZE[dataset_var]
        else:
            raise ValueError(f'Invalid dataset: {dataset}')

        self.parse_fp = join(dp_parsed, f'{dataset_var}.json')
        
        if reveal_num_mode == 'n_abs_non':
            mask_fn = f'{dataset_var}-rand-reveal_n_abs_non'
        elif reveal_num_mode == 'ratio':
            mask_fn = f'{dataset_var}-rand-reveal_{reveal_ratio}'
        else:
            raise ValueError(f'Invalid reveal_num_mode: {reveal_num_mode}')
            
        if self.verb_as_slot:
            mask_fn += '-verb_as_slot'

        if not self.rand_expose:
            mask_fn += '-no_rand_expose'

        self.mask_fp = join(dp_mask, f'{mask_fn}.json')
        self.masked_dicts = self.mask()
    
    def get_summary_obj(self, line):
        summary_obj = SummaryObject(line=line, 
                                    rand_expose=self.rand_expose, 
                                    verb_as_slot=self.verb_as_slot,
                                    reveal_num_mode=self.reveal_num_mode,
                                    reveal_ratio=self.reveal_ratio)
        return summary_obj

    def get_record(self, summary_obj):
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
            for line in tqdm(f, total=self.n_summary):
                # print('===========================Start a new summary==========')
                summary_obj = self.get_summary_obj(line=line.rstrip('\n'))

                n_sentences_wo_init_non_abs += len([so for so in summary_obj.sentence_objects if so.no_init_abs_non_span])
                n_sentences += len(summary_obj.sentence_objects)

                summary_obj.reveal()
                masked_dict = self.get_record(summary_obj)
                masked_dicts.append(masked_dict)
        
        self.n_sentences_wo_init_non_abs = n_sentences_wo_init_non_abs
        self.n_sentences = n_sentences
        return masked_dicts
        
    def dump_mask(self):
        with open(self.mask_fp, 'a') as f:
            for md in self.masked_dicts:
                f.write(json.dumps(md, ensure_ascii=False)+'\n')
        print(f'{len(self.masked_dicts)} masked records have been dumped to {self.mask_fp}')


def parse(dataset, dataset_var, start=0, end=-1):
    if dataset == 'mn':
        s_parser = SummaryParser(dataset_var=dataset_var)
        s_parser.parse_and_dump()

    elif dataset == 'cnndm':
        s_parser = CnnDmSummaryParser(dataset_var=dataset_var)
        s_parser.parse_and_dump(start=start, end=end)
    else:
        raise ValueError(f'Invalid dataset: {dataset}')


def mask(dataset, dataset_var, rand_expose, verb_as_slot, reveal_num_mode, reveal_ratio):
    s_masker = SummaryMasker(dataset=dataset,
        dataset_var=dataset_var, 
        rand_expose=rand_expose, 
        verb_as_slot=verb_as_slot,
        reveal_num_mode=reveal_num_mode,
        reveal_ratio=reveal_ratio)
    s_masker.dump_mask()


def init_non_abs_stats(dataset, dataset_var, rand_expose, verb_as_slot):
    s_masker = SummaryMasker(dataset=dataset,
        dataset_var=dataset_var, 
        rand_expose=rand_expose, 
        verb_as_slot=verb_as_slot)

    print(f'n_sentences_wo_init_non_abs: {s_masker.n_sentences_wo_init_non_abs}, n_sentences: {s_masker.n_sentences}, \
        ratio: {s_masker.n_sentences_wo_init_non_abs/s_masker.n_sentences}')
   

if __name__ == "__main__":
    dataset = 'mn'  # cnndm, mn
    dataset_var = 'val'
    start = 0
    end = -1
    # parse(dataset=dataset, dataset_var=dataset_var, start=start, end=end)
    
    params = {
        'dataset': dataset,  # cnndm, mn,
        'dataset_var': dataset_var,
        'rand_expose': True,
        'verb_as_slot': False,
        'reveal_num_mode': 'ratio',  # n_abs_non or ratio
        'reveal_ratio': 0.85,
    }
    mask(**params)
    # init_non_abs_stats(**params)
