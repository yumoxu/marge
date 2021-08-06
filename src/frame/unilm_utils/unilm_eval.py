# -*- coding: utf-8 -*-
import sys
from os.path import isfile, isdir, join, dirname, abspath, exists

sys_path = dirname(dirname(abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

parent_sys_path = dirname(sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

parent_sys_path = dirname(parent_sys_path)
if parent_sys_path not in sys.path:
    sys.path.insert(0, parent_sys_path)

import os
import io
from pathlib import Path
import copy
import json
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

import utils.tools as tools
from utils.config_loader import path_parser
import summ.compute_rouge as rouge
# import querysum.bert_marge.marge_config as marge_config


class UniLMEval:
    def __init__(self, marge_config, pre_tokenize_sent, max_eval_len, cluster_ids, 
        eval_mn=False, eval_tdqfs=False):
        """
            marge_config should define:
            - UNILM_OUT_DIR_NAME
            - UNILM_DECODE_FILE_PATH

        """
        super().__init__()
        self.marge_config = marge_config
        self._tok_dict = {}
        self.pre_tokenize_sent = pre_tokenize_sent
        self.max_eval_len = max_eval_len
        self.out_dp = path_parser.unilm_out / self.marge_config.UNILM_OUT_DIR_NAME

        self.cids = cluster_ids
        self.eval_mn = eval_mn
        self.eval_tdqfs = eval_tdqfs

    def _is_digit(self, w):
        for ch in w:
            if not(ch.isdigit() or ch == ','):
                return False
        return True

    def fix_tokenization(self, text):
        import string
        input_tokens = text.split()
        output_tokens = []
        i = 0
        prev_dash = False
        while i < len(input_tokens):
            tok = input_tokens[i]
            flag_prev_dash = False
            if tok in self._tok_dict.keys():
                output_tokens.append(self._tok_dict[tok])
                i += 1
            elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
                output_tokens[-1] = output_tokens[-1][:-1]
                output_tokens.append("n't")
                i += 2
            elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
                output_tokens.append("'"+input_tokens[i + 1])
                i += 2
            elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
                output_tokens.append("...")
                i += 3
            elif tok == "," and len(output_tokens) > 0 and self._is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and self._is_digit(input_tokens[i + 1]):
                # $ 3 , 000 -> $ 3,000
                output_tokens[-1] += ','+input_tokens[i + 1]
                i += 2
            elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
                # 3 . 03 -> $ 3.03
                output_tokens[-1] += '.'+input_tokens[i + 1]
                i += 2
            elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
                # U . N . -> U.N.
                k = i+3
                while k+2 < len(input_tokens):
                    if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                        k += 2
                    else:
                        break
                output_tokens[-1] += ''.join(input_tokens[i:k])
                i += 2
            elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
                output_tokens[-1] += tok
                i += 1
            else:
                output_tokens.append(tok)
                i += 1
            prev_dash = flag_prev_dash
        return " ".join(output_tokens)

    def eval_unilm_out_archive(self):
        # cids = tools.get_test_cc_ids()
        out_dp = path_parser.unilm_out / self.marge_config.UNILM_OUT_DIR_NAME
        assert not exists(out_dp), f'{out_dp} exists. Remove it before creating a new one!'
        os.mkdir(out_dp)

        lines = io.open(self.marge_config.UNILM_DECODE_FILE_PATH).readlines()
        lines = [ll.strip('\n') for ll in lines]
        import nltk

        for idx, cid in enumerate(self.cids):
            dec = lines[idx]
            if self.pre_tokenize_sent:
                dec_lines = nltk.tokenize.sent_tokenize(dec)
                dec = '\n'.join(dec_lines)
            with open(join(out_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(dec)
        
        performance = rouge.compute_rouge_abs(out_dp, 
            split_sentences=not self.pre_tokenize_sent,
            eval_mn=self.eval_mn)
        return performance

    def eval_unilm_out(self):
        assert exists(self.out_dp), f'Build unilm_out before evaluating it: {self.out_dp}'

        if not self.eval_tdqfs:
            performance = rouge.compute_rouge_abs_f1(self.out_dp, 
                split_sentences=not self.pre_tokenize_sent, 
                max_len=self.max_eval_len,
                eval_mn=self.eval_mn)
        else:
            performance = rouge.compute_rouge_for_tdqfs(
                text_dp=self.out_dp,
                length=self.max_eval_len,
                ref_dp=path_parser.data_tdqfs_summary_targets)
        return performance

    def build_and_eval_unilm_out(self):
        # cids = tools.get_test_cc_ids()
        assert not exists(self.out_dp), f'{self.out_dp} exists. Remove it before creating a new one!'
        os.mkdir(self.out_dp)

        lines = io.open(self.marge_config.UNILM_DECODE_FILE_PATH).readlines()
        lines = [ll.strip('\n') for ll in lines]
        import nltk

        for idx, cid in enumerate(self.cids):
            dec = lines[idx].strip()
            fixed_dec = self.fix_tokenization(dec) 
            print(f'dec: {dec}\nfixed_dec: {fixed_dec}')
            while "  " in fixed_dec:
                fixed_dec = fixed_dec.replace("  ", " ")
            
            if self.pre_tokenize_sent:
                dec_lines = nltk.tokenize.sent_tokenize(fixed_dec)
                fixed_dec = '\n'.join(dec_lines)
            
            with open(join(self.out_dp, cid), mode='a', encoding='utf-8') as out_f:
                out_f.write(fixed_dec)
        
        return self.eval_unilm_out()


class UniLMEvalSubQ:
    def __init__(self, marge_config, pre_tokenize_sent, max_eval_len, 
            cluster_ids, uids, integrate_mode,
            eval_mn=False):
        """
            marge_config should define:
            - UNILM_OUT_DIR_NAME
            - UNILM_DECODE_FILE_PATH

        """
        super().__init__()
        self.marge_config = marge_config
        self._tok_dict = {}
        self.pre_tokenize_sent = pre_tokenize_sent
        self.max_eval_len = max_eval_len
        self.out_dp = path_parser.unilm_out / self.marge_config.UNILM_OUT_DIR_NAME

        self.cids = cluster_ids
        self.uids = uids

        self.integrate_mode = integrate_mode
        self.integrate_dp = join(self.out_dp, self.integrate_mode)

        self.eval_mn = eval_mn

    def _is_digit(self, w):
        for ch in w:
            if not(ch.isdigit() or ch == ','):
                return False
        return True

    def fix_tokenization(self, text):
        import string
        input_tokens = text.split()
        output_tokens = []
        i = 0
        prev_dash = False
        while i < len(input_tokens):
            tok = input_tokens[i]
            flag_prev_dash = False
            if tok in self._tok_dict.keys():
                output_tokens.append(self._tok_dict[tok])
                i += 1
            elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
                output_tokens[-1] = output_tokens[-1][:-1]
                output_tokens.append("n't")
                i += 2
            elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
                output_tokens.append("'"+input_tokens[i + 1])
                i += 2
            elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
                output_tokens.append("...")
                i += 3
            elif tok == "," and len(output_tokens) > 0 and self._is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and self._is_digit(input_tokens[i + 1]):
                # $ 3 , 000 -> $ 3,000
                output_tokens[-1] += ','+input_tokens[i + 1]
                i += 2
            elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
                # 3 . 03 -> $ 3.03
                output_tokens[-1] += '.'+input_tokens[i + 1]
                i += 2
            elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
                # U . N . -> U.N.
                k = i+3
                while k+2 < len(input_tokens):
                    if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                        k += 2
                    else:
                        break
                output_tokens[-1] += ''.join(input_tokens[i:k])
                i += 2
            elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
                output_tokens[-1] += tok
                i += 1
            else:
                output_tokens.append(tok)
                i += 1
            prev_dash = flag_prev_dash
        return " ".join(output_tokens)

    def build_subq_out(self):
        assert not exists(self.out_dp), f'{self.out_dp} exists. Remove it before creating a new one!'
        os.mkdir(self.out_dp)

        lines = io.open(self.marge_config.UNILM_DECODE_FILE_PATH).readlines()
        lines = [ll.strip('\n') for ll in lines]
        
        for idx, uid in enumerate(self.uids):
            dec = lines[idx].strip()
            fixed_dec = self.fix_tokenization(dec) 
            print(f'dec: {dec}\nfixed_dec: {fixed_dec}')
            while "  " in fixed_dec:
                fixed_dec = fixed_dec.replace("  ", " ")
            
            assert not self.pre_tokenize_sent, 'We put different summaries in the same file so one line is one summary.'
            # if self.pre_tokenize_sent:
                # dec_lines = nltk.tokenize.sent_tokenize(fixed_dec)
                # fixed_dec = '\n'.join(dec_lines)
            
            cid = '_'.join(uid.split('_')[:-1])
            with open(join(self.out_dp, f'{cid}_subq'), mode='a', encoding='utf-8') as out_f:
                out_f.write(fixed_dec+'\n')

    def _integrate(self, subq_summ):
        if len(subq_summ) == 1:
            return subq_summ[0]

        assert self.integrate_mode == 'uniform', f'Not implemented integrate_mode: {self.integrate_mode}'
        avai_budget = self.max_eval_len
        avai_n_subq = len(subq_summ)
        final_sents = []

        for _summ in subq_summ:
            curr_budget = avai_budget / avai_n_subq
            subq_sents = nltk.tokenize.sent_tokenize(_summ)
            subq_nw = [len(nltk.tokenize.word_tokenize(ss)) for ss in subq_sents]
            accumulated_nw = 0

            for sent_idx, sent_nw in enumerate(subq_nw):
                accumulated_nw += sent_nw
                if accumulated_nw >= curr_budget:
                    final_sents.extend(subq_sents[:sent_idx])
                    break

            avai_budget -= accumulated_nw
            avai_n_subq -= 1
        return ' '.join(final_sents)

    def integrate_subq_out(self):
        assert os.exists(self.integrate_dp), f'Remove the existing integrate_dp: {integrate_dp}'
        os.mkdir(self.integrate_dp)

        for cid in self.cids:
            subq_fn = f'{cid}_subq'
            subq_summ = [line.strip('\n') for line in open(join(self.out_dp, subq_fn), encoding='utf-8').readlines()]
            subq_summ = [line for line in subq_summ if line]
            
            integrate_summ = self._integrate(subq_summ)
            open(join(integrate_dp, cid), mode='a', encoding='utf-8').write(integrate_summ)
    
    def eval_unilm_out(self):
        assert exists(self.integrate_dp), f'Build integrate_dp before evaluating it: {self.integrate_dp}'

        performance = rouge.compute_rouge_abs(self.integrate_dp, 
            split_sentences=not self.pre_tokenize_sent, 
            max_len=self.max_eval_len,
            eval_mn=self.eval_mn)
        return performance

    def build_and_eval_unilm_out(self):
        self.build_subq_out()
        self.integrate_subq_out()
        
        self.eval_unilm_out()
