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
    def __init__(self, open_ie_dict, rand_expose: bool, verb_as_slot: bool, max_rand_trials=100, 
            mask_token='[MASK]',
            base_token='[BASE]',
            remove_base=None,
            func_reveal_ratio=1.0):
        """
            open_ie_dict: a dictionary storing OpenIE parsing results
            rand_expose: when init non_fids=None
            verb_as_slot: whether include verbs as slots
            func_reveal_ratio: sample from abs_non_span
        """

        self.raw_sentence = open_ie_dict['raw_sentence']
        self.words = open_ie_dict['words']
        self.n_words = len(self.words)

        self.facts = [Fact(tags=verb_dict['tags'], verb_as_slot=verb_as_slot) for verb_dict in open_ie_dict['verbs']]

        self.rand_expose = rand_expose
        self.max_rand_trials = max_rand_trials
        self.mask_token = mask_token
        self.base_token = base_token
        self.remove_base = remove_base
        self.func_reveal_ratio = func_reveal_ratio
        
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

    def init_masked_words(self):
        """
            Fill words/a label to  abs_non_slot.
            
        """
        abs_non_span = self.get_abs_non_span()
        n_abs_non = len(abs_non_span)
        
        if self.func_reveal_ratio < 1.0 and n_abs_non > 1:  # sample from abs_non_span
            n_func_reveal = int(n_abs_non * self.func_reveal_ratio)
            abs_non_span = random.sample(abs_non_span, n_func_reveal) if n_func_reveal>0 else []

        masked_words = [self.mask_token] * len(self.words)  # init masked words
        
        if self.remove_base:
            self.fill(indices=abs_non_span, src=self.words, target=masked_words, fill_token=self.base_token)
        else:
            self.fill(indices=abs_non_span, src=self.words, target=masked_words)
        
        return masked_words, len(abs_non_span), n_abs_non

    def get_available_facts(self):
        available_facts = [fact for fact in self.facts if fact.is_available()]
        return available_facts

    def is_available(self):
        """
            # A sentence is available when it has at least one available fact.
            A sentence is available when it has at least one available mask token.
        """
        # available_facts = [for token in ]
        # if self.n_revealed < self.n_words:
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

    def fill(self, indices, src, target, fill_token=None):
        """
            Fill target with src at the given indices.

            n_filled: number of indices newly filled.
        """
        n_filled = 0
        for idx in indices:
            if target[idx] == self.mask_token:
                if not fill_token:
                    target[idx] = src[idx]
                else:
                    target[idx] = fill_token
                
                n_filled += 1
        return n_filled

    def reveal(self):
        """
            Reveal a random fact.
            Return False if no eligible fact is revealed.

            Since selected slot could bring no gains to the current masked sequence, we sample *max_rand_trials* times to get an eligible slot.

        """
        n_trial = 0
        while n_trial < self.max_rand_trials:
            n_trial += 1
            if not self.is_available():
                return False
            
            slot, n_ticks = self.rand_slot()
            n_filled = self.fill(indices=slot.span, src=self.words, target=self.masked_words)
            if not n_filled:
                if n_trial % 20 == 0:
                    available_slot_ids = [fc.get_available_slot_ids() for fc in self.get_available_facts()]
                    print(f'n_trial: {n_trial}, n_ticks: {n_ticks}, masked_words: {self.masked_words}, #available facts: {len(self.get_available_facts())}, available slots: {available_slot_ids}')
                continue
            
            self.n_revealed += n_filled
            return True

        return False

    def merge(self):
        if self.remove_base:
            self.masked_words = [word for word in self.masked_words if word != self.base_token]
            token_set = list(set(self.masked_words))
            if len(token_set) == 1 and token_set[0] == self.mask_token:
                self.merged_masked_words = []
                return
        
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
            min_nw_mode: str, 
            reveal_ratio=1.0,
            max_rand_trials=100, 
            mask_token='[MASK]', 
            base_token='[BASE]',
            sub_token='[SUBQUERY]',
            remove_base=None,
            max_nw=None,
            func_reveal_ratio=1.0):
        """
            Inputs: 
                line: a json string
                rand_expose: when init non_fids=None
                verb_as_slot: whether include verbs as slots
                min_nw_mode: 'sample', 'max', 'ratio'
                reveal_ratio: applied to the total nw of the summary; max: 1.0
                max_rand_trials: 
                open_ie_dict: a dictionary storing OpenIE parsing results
                mask_token: 
                max_nw: 
                remove_base: whether to remove base reveals (abs non span words) for the final seq

            Attributes:
                n_sentences: #sentences in summary
                nw: total #words in summary
                min_nw: min #words for slot revealing
                max_nw: max #words for final clipping
                merged_masked_words: 2D list; masked words for each sentence (after merging)
                masked_seq: 1D list; a flat version of merged_masked_words.
        """
        self.rand_expose = rand_expose
        self.verb_as_slot = verb_as_slot
        # self.min_nw_mode = min_nw_mode
        self.max_rand_trials = max_rand_trials
        self.mask_token = mask_token
        self.base_token = base_token
        self.remove_base = remove_base
        self.func_reveal_ratio = func_reveal_ratio
        
        summary_info = json.loads(line)
        self.raw_summary = summary_info['raw_summary']
        self.sentence_objects = self.get_sentence_objects(open_ie_dicts=summary_info['open_ie'])        
        self.n_sentences = len(self.sentence_objects)

        self.nw = sum([so.n_words for so in self.sentence_objects])
        # reveal_base = self.get_global_n_revealed()
        # assert 0.0 <= reveal_ratio <= 1.0
        # self.min_nw = int(reveal_base + (self.nw - reveal_base) * reveal_ratio)  # exclude base from ratio
        # print(f'Init: nw: {self.nw}: reveal_base: {reveal_base}, min_nw: {self.min_nw}')
        
        reveal_base = self.get_global_n_revealed()
        global_n_abs_non = self.get_global_n_abs_non()
        n_inf_words = self.nw - global_n_abs_non

        assert 0.0 <= reveal_ratio <= 1.0
        self.min_nw = int(reveal_base + n_inf_words * reveal_ratio)  # exclude global_n_abs_non from ratio
        print(f'Init: nw: {self.nw}: reveal_base: {reveal_base}, global_n_abs_non: {global_n_abs_non}, min_nw: {self.min_nw}')

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
            'max_rand_trials': self.max_rand_trials,
            'mask_token': self.mask_token,
            'base_token': self.base_token,
            'remove_base': self.remove_base,
            'func_reveal_ratio': self.func_reveal_ratio,
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

    def get_available_sentence_objects(self):
        available_sentence_objects = [so for so in self.sentence_objects if so.is_available()]
        return available_sentence_objects
        
    def reveal(self, available_sentence_objects):
        """
            Return True if revealed words exceeds min_nw, otherwise False.
        """
        for so in available_sentence_objects:
            reveal_sucess = so.reveal()
            if not reveal_sucess:
                continue

            global_n_revealed = self.get_global_n_revealed()
            # print(f'Reveal: global_n_revealed: {global_n_revealed}, min_nw: {self.min_nw}')

            if global_n_revealed >= self.min_nw:
                return True
        
        return False

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
                # since more tokens may be inserted and the final ength is not set
                # we leave the last [SEP] to later BERT input proc 
                self.masked_seq_with_sep += [self.sep_token]
                self.masked_seq_with_sub += [self.sub_token]
        
        if self.max_nw:
            self.masked_seq = self.masked_seq[:self.max_nw]
            self.masked_seq_with_sep = self.masked_seq_with_sep[:self.max_nw]
            self.masked_seq_with_sub = self.masked_seq_with_sub[:self.max_nw]
        
    def iteratively_reveal(self):
        """
            Return n_iters when iterative revealing is done.

            Run get_available_sentence_objects after every iteration of all summary sentences.
        """
        n_iters = 0
        if self.min_nw < 0:  # the init #reveals has already exceeded masimum #words.
            self.finalize()
            return n_iters
        
        excess = False
        while not excess and n_iters < 100:
            n_iters += 1
            available_sentence_objects = self.get_available_sentence_objects()
            if not available_sentence_objects:
                # raise ValueError('Failed to reveal slots in summary: no available sentence objects.')
                self.finalize()
                return n_iters

            excess = self.reveal(available_sentence_objects)
            if excess:
                self.finalize()
                return n_iters
        
        raise ValueError('Failed to reveal slots in summary: achieved the maximum #iters.')


class SummaryMasker:
    def __init__(self, dataset, dataset_var, 
            rand_expose, 
            verb_as_slot, 
            min_nw_mode: str, 
            reveal_ratio=1.0,
            max_rand_trials=100, 
            mask_token='[MASK]',
            remove_base=None,
            func_reveal_ratio=1.0):
        self.dataset_var = dataset_var
        self.rand_expose = rand_expose
        self.verb_as_slot = verb_as_slot
        self.max_rand_trials = max_rand_trials
        self.mask_token = mask_token
        self.min_nw_mode = min_nw_mode
        self.reveal_ratio = reveal_ratio
        self.remove_base = remove_base
        self.func_reveal_ratio = func_reveal_ratio

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
        mask_fn = f'{dataset_var}-{min_nw_mode}-reveal_{self.reveal_ratio}'
        if self.verb_as_slot:
            mask_fn += '-verb_as_slot'
        
        if self.remove_base:
            mask_fn += '-remove_base'

        if not self.rand_expose:
            mask_fn += '-no_rand_expose'
        
        if func_reveal_ratio < 1.0:
            mask_fn += f'-func_reveal_{func_reveal_ratio}'

        self.mask_fp = join(dp_mask, f'{mask_fn}.json')
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

                summary_obj.iteratively_reveal()
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


def mask(dataset, dataset_var, rand_expose, verb_as_slot, min_nw_mode, reveal_ratio, remove_base, func_reveal_ratio):
    s_masker = SummaryMasker(dataset=dataset,
        dataset_var=dataset_var, 
        rand_expose=rand_expose, 
        verb_as_slot=verb_as_slot, 
        min_nw_mode=min_nw_mode,
        reveal_ratio=reveal_ratio,
        max_rand_trials=100,
        remove_base=remove_base,
        func_reveal_ratio=func_reveal_ratio)
    s_masker.dump_mask()


def init_non_abs_stats(dataset, dataset_var, rand_expose, verb_as_slot, min_nw_mode, reveal_ratio, remove_base, func_reveal_ratio):
    s_masker = SummaryMasker(dataset=dataset,
        dataset_var=dataset_var, 
        rand_expose=rand_expose, 
        verb_as_slot=verb_as_slot, 
        min_nw_mode=min_nw_mode,
        reveal_ratio=reveal_ratio,
        max_rand_trials=100,
        remove_base=remove_base,
        func_reveal_ratio=func_reveal_ratio)

    print(f'n_sentences_wo_init_non_abs: {s_masker.n_sentences_wo_init_non_abs}, n_sentences: {s_masker.n_sentences}, \
        ratio: {s_masker.n_sentences_wo_init_non_abs/s_masker.n_sentences}')
    
    
def _draw(n_token, range, n_bins, xlabel, title, color='darkblue'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_token = [int(nt) for nt in n_token]
    counts, bin_edges = np.histogram(n_token, bins=n_bins, range=range, density=False)
    dist = [c/float(sum(counts)) for c in counts]

    logger.info(f'total counts: {sum(counts)}')
    logger.info(f'distribution: {dist}')
    logger.info(f'bin_edges: {bin_edges}')

    fig = plt.figure(figsize=(5, 4))
    sns.distplot(n_token, hist=True, kde=True, 
        bins=n_bins, 
        color=color, 
        hist_kws={'edgecolor':'black', 'range': range}, 
        kde_kws={'linewidth': 2})
    # plt.hist(np.array(n_words), bins=7, range=(10, 80), density=True, stacked=True)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.tight_layout()
    
    fig.savefig(DP_PROJ/'stats'/ title, bbox_inches='tight')
    plt.show()


def mask_summary_stats(dataset, 
        dataset_var, 
        rand_expose,
        verb_as_slot, 
        min_nw_mode, 
        reveal_ratio, 
        remove_base, 
        func_reveal_ratio,
        token_seq_key='masked_seq'):
    """
        Compute #token and #[MASK] in masked summaries.
        
        Draw and dump the distributions.
    """
    mask_fn = f'{dataset_var}-{min_nw_mode}-reveal_{reveal_ratio}'
    
    if verb_as_slot:
        mask_fn += '-verb_as_slot'
    
    if remove_base:
        mask_fn += '-remove_base'
    
    if not rand_expose:
        mask_fn += '-no_rand_expose'

    if func_reveal_ratio < 1.0:
        mask_fn += f'-func_reveal_{func_reveal_ratio}'
    
    if dataset == 'mn':
        mask_fp = DP_MASKED_MN_SUMMARY / f'{mask_fn}.json'
    else:
        mask_fp = DP_MASKED_CNNDM_SUMMARY / f'{mask_fn}.json'
    # n_summary = CNNDM_DATASET_SIZE[dataset_var]
    
    stat_title = mask_fn if dataset == 'mn' else f'{dataset}-{mask_fn}'
    stat_fp = DP_PROJ / 'stats' / f'{stat_title}.txt'

    if not exists(stat_fp):
        logger.info('Build stat file')
        with io.open(stat_fp, mode='a') as stat_f:
            stat_f.write('n_token\tn_mask\n')
            with io.open(mask_fp) as mask_f:
                lines = mask_f.readlines()
                for line in tqdm(lines):
                    masked_seq = json.loads(line.strip('\n'))[token_seq_key]
                    n_token = len(masked_seq)
                    n_mask = masked_seq.count('[MASK]')
                    stat_f.write(f'{n_token}\t{n_mask}\n')
    
    logger.info(f'Read from stat file: {stat_fp}')
    with io.open(stat_fp) as stat_f:
        lines = stat_f.readlines()[1:]
        items = [line.strip('\n').split('\t') for line in lines]
        n_token, n_mask = zip(*items)
        
        _draw(n_token=n_token, range=[10, 250], n_bins=24, color='darkblue',
            xlabel='Number of words', title=f'token_dist_{stat_title}.pdf')
        _draw(n_token=n_mask, range=[0, 50], n_bins=50, color='darkred',
            xlabel='Number of [MASK] tokens', title=f'mask_dist_{stat_title}.pdf')


def mask_summary_avg(dataset, 
        dataset_var, 
        rand_expose,
        verb_as_slot, 
        min_nw_mode, 
        reveal_ratio, 
        remove_base, 
        func_reveal_ratio,
        token_seq_key='masked_seq'):
    mask_fn = f'{dataset_var}-{min_nw_mode}-reveal_{reveal_ratio}'
    
    if verb_as_slot:
        mask_fn += '-verb_as_slot'
    
    if remove_base:
        mask_fn += '-remove_base'
    
    if not rand_expose:
        mask_fn += '-no_rand_expose'

    if func_reveal_ratio < 1.0:
        mask_fn += f'-func_reveal_{func_reveal_ratio}'
    
    if dataset == 'mn':
        mask_fp = DP_MASKED_MN_SUMMARY / f'{mask_fn}.json'
    else:
        mask_fp = DP_MASKED_CNNDM_SUMMARY / f'{mask_fn}.json'
    
    stat_title = mask_fn if dataset == 'mn' else f'{dataset}-{mask_fn}'
    stat_fp = DP_PROJ / 'stats' / f'{stat_title}.txt'

    if not exists(stat_fp):
        logger.info('Build stat file')
        with io.open(stat_fp, mode='a') as stat_f:
            stat_f.write('n_token\tn_mask\n')
            with io.open(mask_fp) as mask_f:
                lines = mask_f.readlines()
                for line in tqdm(lines):
                    masked_seq = json.loads(line.strip('\n'))[token_seq_key]
                    n_token = len(masked_seq)
                    n_mask = masked_seq.count('[MASK]')
                    stat_f.write(f'{n_token}\t{n_mask}\n')
    
    logger.info(f'Read from stat file: {stat_fp}')
    with io.open(stat_fp) as stat_f:
        lines = stat_f.readlines()[1:]
        items = [line.strip('\n').split('\t') for line in lines]
        n_token, n_mask = zip(*items)
        avg_token = sum([int(item) for item in n_token]) / len(n_token)
        avg_mask = sum([int(item) for item in n_mask]) / len(n_mask)

        print(f'avg_token: {avg_token}, avg_mask: {avg_mask}')

    
if __name__ == "__main__":
    dataset = 'mn'  # cnndm, mn
    dataset_var = 'train'
    start = 0
    end = -1
    # parse(dataset=dataset, dataset_var=dataset_var, start=start, end=end)
    
    params = {
        'dataset': dataset,  # cnndm, mn,
        'dataset_var': dataset_var,
        'rand_expose': True,
        'verb_as_slot': False,
        'min_nw_mode': 'ratio',  # max, sample
        'reveal_ratio': 0.0,
        'remove_base': False,
        'func_reveal_ratio': 1.0,
    }
    # mask(**params)
    # init_non_abs_stats(**params)
    
    if dataset_var == 'train':
        token_seq_key = 'masked_seq'  # masked_seq_with_sub
        # mask_summary_stats(**params, token_seq_key=token_seq_key)
        mask_summary_avg(**params, token_seq_key=token_seq_key)
