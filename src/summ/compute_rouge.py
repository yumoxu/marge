import os
from os.path import join, dirname, abspath, exists
import sys

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from pyrouge import Rouge155
import utils.config_loader as config
from utils.config_loader import config_meta, path_parser, logger
import utils.tools as tools
from pathlib import Path
import logging

def get_rouge_2_recall(output):
    start_pat = '1 ROUGE-2 Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    target_ck = None
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            if ck.startswith(start_pat):
                target_ck = ck
                break

    if not target_ck:
        raise ValueError('Not found record!')

    num_idx = 3
    lines = target_ck.split('\n')
    recall = float(lines[0].split(' ')[num_idx])

    return recall


def compute_rouge_2_recall_for_multinews_sentence(sentence, summary, temp_dir: Path):
    """
        sentence: target multinews sentence
        summary: reference summary to compute ROUGE against
        temp_dir: temp dir for computing ROUGE
        save_fp: save fp for ROUGE-2 results: Recall\tF1
    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {} -x'.format(path_parser.afs_rouge_dir)
    else:
        rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval

    sent_dp = temp_dir / 'sentence'
    summary_dp = temp_dir / 'summary'
    
    if not exists(sent_dp):
        os.mkdir(sent_dp)
    
    if not exists(summary_dp):
        os.mkdir(summary_dp)

    sent_fp =  sent_dp / 'temp'
    summary_fp = summary_dp / 'temp'
    open(sent_fp, 'a').write(sentence)
    open(summary_fp, 'a').write(summary)
        
    # r = Rouge155(rouge_args=rouge_args)
    r = Rouge155(rouge_dir='~/pyrouge/RELEASE-1.5.5', rouge_args=rouge_args)
    r.system_dir = sent_dp
    r.model_dir = summary_dp

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    recall = get_rouge_2_recall(output)
    os.remove(sent_fp)
    os.remove(summary_fp)
    
    return recall


def compute_rouge_for_mn(text_dp, budget=None):
    """

    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'  # *todo*: double check -l 300
        if budget:
            rouge_args += f'-l {budget}'
        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        if budget:
            rouge_args += f'-l {budget}'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    r.system_dir = text_dp
    r.model_dir = path_parser.data_mn_summary_targets

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output)
    logger.info(output)
    return output

"""
    Codes belows is borrowed from QuerySum codebase.
"""

def proc_output_for_tune_recall(output):
    start_pat = '1 ROUGE-2 Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    target_ck = None
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            if ck.startswith(start_pat):
                target_ck = ck
                break

    if not target_ck:
        raise ValueError('Not found record!')

    num_idx = 3
    lines = target_ck.split('\n')
    recall = '{0:.2f}'.format(float(lines[0].split(' ')[num_idx]) * 100)
    f1 = '{0:.2f}'.format(float(lines[2].split(' ')[num_idx]) * 100)

    return '\t'.join((recall, f1))


def proc_output_for_tune(output):
    target = ['1', '2']
    start_pat = '1 ROUGE-{} Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    tg2ck = {}
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            for tg in target:
                if ck.startswith(start_pat.format(tg)):
                    tg2ck[tg] = ck
                    break

    num_idx = 3
    tg2f1 = {}
    for tg, ck in tg2ck.items():
        lines = ck.split('\n')
        f1 = '{0:.2f}'.format(float(lines[2].split(' ')[num_idx]) * 100)
        tg2f1[tg] = f1

    return '\t'.join([tg2f1[tg] for tg in target])


def proc_output(output):
    target = ['1', '2', 'SU4']
    start_pat = '1 ROUGE-{} Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    tg2ck = {}
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            for tg in target:
                if ck.startswith(start_pat.format(tg)):
                    tg2ck[tg] = ck
                    break

    num_idx = 3

    tg2recall = {}
    tg2f1 = {}

    for tg, ck in tg2ck.items():
        lines = ck.split('\n')
        recall = '{0:.2f}'.format(float(lines[0].split(' ')[num_idx]) * 100)
        f1 = '{0:.2f}'.format(float(lines[2].split(' ')[num_idx]) * 100)

        tg2recall[tg] = recall
        tg2f1[tg] = f1

    recall_str = 'Recall:\t{}'.format('\t'.join([tg2recall[tg] for tg in target]))
    f1_str = 'F1:\t{}'.format('\t'.join([tg2f1[tg] for tg in target]))

    output = '\n' + '\n'.join((f1_str, recall_str))
    return output


def proc_output_f1(output):
    """
        Get only F1 scores.
    """
    target = ['1', '2', 'SU4']
    start_pat = '1 ROUGE-{} Average'

    output = '\n'.join(output.split('\n')[1:])
    inter_breaker = '\n---------------------------------------------\n'
    intra_breaker = '\n.............................................\n'

    tg2ck = {}
    for ck in output.split(inter_breaker):
        ck = ck.strip('\n')
        if ck:
            ck = ck.split(intra_breaker)[0]
            for tg in target:
                if ck.startswith(start_pat.format(tg)):
                    tg2ck[tg] = ck
                    break

    num_idx = 3
    tg2f1 = {}

    for tg, ck in tg2ck.items():
        lines = ck.split('\n')
        f1 = '{0:.2f}'.format(float(lines[2].split(' ')[num_idx]) * 100)

        tg2f1[tg] = f1

    f1_str = '\t'.join([tg2f1[tg] for tg in target])
    return f1_str


def compute_rouge_mix(model_name, n_iter, cos_threshold, extra):
    for year in config.years:
        compute_rouge(model_name, n_iter=n_iter,
                      cos_threshold=cos_threshold,
                      year=year,
                      extra=extra)
    return None


def compute_rouge_for_dev(text_dp, tune_centrality):
    """
        This function is for tuning paramters.

        It has been revised according to margesum environments.

    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'
        if tune_centrality:  # summary length requirement
            rouge_args += ' -l 250'

        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        if tune_centrality:  # summary length requirement
            rouge_args += ' -l 250'
        
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    r.system_dir = text_dp
    r.model_dir = join(path_parser.data_summary_targets, config.test_year)

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output_for_tune(output)
    logger.info(output)
    return output


def compute_rouge_for_cont_sel_in_sentences(text_dp):
    """
        This function is for tuning paramters.

        It has been revised according to margesum environments.

    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'

        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    r.system_dir = text_dp
    r.model_dir = join(path_parser.data_summary_targets, config.test_year)

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output_for_tune_recall(output)
    logger.info(output)
    return output


def compute_rouge_for_cont_sel_in_sentences_tdqfs(text_dp, ref_dp):
    """
        This function is for tuning paramters.

        It has been revised according to margesum environments.

    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'

        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    r.system_dir = text_dp
    r.model_dir = ref_dp

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output_for_tune_recall(output)
    logger.info(output)
    return output


def compute_rouge_for_cont_sel(text_dp, n_words):
    """
        This function is for tuning paramters.

        It has been revised according to margesum environments.

    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x -l {n_words}'
        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x -l {n_words}'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    r.system_dir = text_dp
    r.model_dir = join(path_parser.data_summary_targets, config.test_year)

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output_for_tune(output)
    logger.info(output)
    return output


def compute_rouge(model_name, n_iter=None, diversity_param_tuple=None, cos_threshold=None, extra=None):
    """
        This function is for calculating final ROUGE scores. 

        It has been revised according to margesum environments.

    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -l 250 -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'
        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -l 250 -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    baselines_wo_config = ['lead', 'lead-2006', 'lead-2007', 'lead_2007']
    if model_name in baselines_wo_config or model_name.startswith('duc'):
        text_dp = join(path_parser.summary_text, model_name)
    else:
        text_dp = tools.get_text_dp(model_name,
                                    cos_threshold=cos_threshold,
                                    n_iter=n_iter,
                                    diversity_param_tuple=diversity_param_tuple,
                                    extra=extra)

    r.system_dir = text_dp
    r.model_dir = join(path_parser.data_summary_targets, config.test_year)
    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'

    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output)
    # output = proc_output_for_tune(output)
    logger.info(output)
    return output


def compute_rouge_for_ablation_study(text_dp):
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = '-a -l 250 -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {} -x'.format(
            path_parser.afs_rouge_dir)

    else:
        rouge_args = '-a -l 250 -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval

    r = Rouge155(rouge_args=rouge_args)
    r.system_dir = text_dp
    r.model_dir = join(path_parser.data_summary_targets, config.test_year)

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output)
    logger.info(output)
    return output


def compute_rouge_abs(text_dp, split_sentences, max_len=250, eval_mn=False):
    """
        eval_mn: determines model_dir
    
    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -l {max_len} -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'

    else:
        rouge_args = f'-a -l {max_len} -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval

    r = Rouge155(rouge_args=rouge_args)
    r.system_dir = text_dp
    r.system_filename_pattern = '(\w*)'
    
    if eval_mn:
        r.model_dir = path_parser.data_mn_summary_targets
        r.model_filename_pattern = '#ID#'
    else:
        r.model_dir = join(path_parser.data_summary_targets, config.test_year)
        r.model_filename_pattern = '#ID#_[\d]'
    
    output = r.convert_and_evaluate(split_sentences=split_sentences)
    output = proc_output(output)
    logger.info(output)
    return output


def compute_rouge_abs_f1(text_dp, split_sentences, max_len=250, eval_mn=False):
    """
        eval_mn: determines model_dir
    
    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -l {max_len} -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'

    else:
        rouge_args = f'-a -l {max_len} -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval

    r = Rouge155(rouge_args=rouge_args)
    r.system_dir = text_dp
    r.system_filename_pattern = '(\w*)'
    
    if eval_mn:
        r.model_dir = path_parser.data_mn_summary_targets
        r.model_filename_pattern = '#ID#'
    else:
        r.model_dir = join(path_parser.data_summary_targets, config.test_year)
        r.model_filename_pattern = '#ID#_[\d]'
    
    output = r.convert_and_evaluate(split_sentences=split_sentences)
    output = proc_output_f1(output)
    return output


def compute_rouge_for_tdqfs(text_dp, ref_dp, length):
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir} -x'
        if length:
            rouge_args += f' -l {length}'
        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir} -x'
        if length:
            rouge_args += f' -l {length}'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    r.system_dir = text_dp
    r.model_dir = ref_dp

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#_[\d]'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output)
    logger.info(output)
    return output


def compute_rouge_for_eli5(text_dp, ref_dp, length):
    """

    """
    if config.path_type == 'afs':  # copied from 2007 with removal of "-l 250"
        rouge_args = f'-a -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.afs_rouge_dir}'
        if length:
            rouge_args += f' -l {length}'
        r = Rouge155(rouge_dir=str(path_parser.remote_root / 'ROUGE-1.5.5'),
                     rouge_args=rouge_args, 
                     log_level=logging.WARNING, 
                     config_parent_dir=str(path_parser.remote_root))
    else:
        # rouge_args = '-a -n 2 -m -c 95 -r 1000 -f A -p 0.5 -t 0 -d -x'  # standard nist eval
        rouge_args = f'-a -n 2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -d -e {path_parser.local_rouge_dir}'
        if length:
            rouge_args += f' -l {length}'
        r = Rouge155(rouge_args=rouge_args, log_level=logging.WARNING)

    r.system_dir = text_dp
    r.model_dir = ref_dp

    gen_sys_file_pat = '(\w*)'
    gen_model_file_pat = '#ID#'
    r.system_filename_pattern = gen_sys_file_pat
    r.model_filename_pattern = gen_model_file_pat

    output = r.convert_and_evaluate()
    output = proc_output(output, target=['1', '2', 'L'])
    logger.info(output)
    return output