import io
import sys
from os import listdir
from os.path import join, dirname, abspath, isfile
import shutil
from shutil import copyfile

import utils.config_loader as config
from utils.config_loader import logger, path_parser
import utils.tools as tools

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def match_summary_fn_with_cid(summary_fn, cid):
    year, cc = cid.split(config.SEP)

    if year == '2005':  # handle cluster naming differences in 2005 data
        summary_fn = summary_fn.lower()

    is_a_match = cc.startswith(summary_fn.split('.')[0])

    return is_a_match


def retrieve_refs_fps_with_cid(cc_ids, ref_dp):
    cid2ref_fps = dict()
    for cid in cc_ids:
        ref_fns = [ref_fn for ref_fn in listdir(ref_dp) if isfile(join(ref_dp, ref_fn))]
        ref_fps = [join(ref_dp, fn) for fn in ref_fns if match_summary_fn_with_cid(fn, cid)]
        cid2ref_fps[cid] = ref_fps

    return cid2ref_fps


def build_summary_targets_single_file(cid2ref_fps, out_dp):
    for cid, ref_fps in cid2ref_fps.items():
        content = []
        for ref_fp in ref_fps:  # handle multiple refs
            summary_words = []
            with io.open(ref_fp, encoding='latin1') as ref_f:
                for line in ref_f:
                    words = config.bert_tokenizer.tokenize(line.rstrip('\n'))
                    summary_words.extend(words)
                # lines = [ll.rstrip('\n') for ll in ref_f.readlines() if ll.rstrip('\n')]
            # content.append(' '.join(lines))
            content.append(' '.join(summary_words))

        out_fp = join(out_dp, cid)
        with io.open(out_fp, mode='a', encoding='utf-8') as out_f:
            out_f.write('\n'.join(content))
        logger.info('[BUILD SUMMARY TARGETS] successfully dumped {0} refs for {1}'.format(len(content), cid))


def build_summary_targets(cid2ref_fps, out_dp, tokenize=True):
    for cid, ref_fps in cid2ref_fps.items():
        for ref_idx, ref_fp in enumerate(ref_fps, start=1):  # handle multiple refs
            # summary_words = []

            out_fp = join(out_dp, config.SEP.join((cid, str(ref_idx))))
            # logger.info('ref_fp: {}'.format(ref_fp))
            # logger.info('out_fp: {}'.format(out_fp))

            if not tokenize:
                shutil.copy(ref_fp, out_fp)
            else:
                summary_sents = []
                with io.open(ref_fp, encoding='latin1') as ref_f:
                    for line in ref_f:
                        words = config.bert_tokenizer.tokenize(line.rstrip('\n'))
                        summary_sents.append(' '.join(words))
                        # summary_words.extend(words)

                ref = '\n'.join(summary_sents)
                # ref = ' '.join(summary_words)
                with io.open(out_fp, mode='a', encoding='utf-8') as out_f:
                    out_f.write(ref)
            # logger.info('[BUILD SUMMARY TARGETS] successfully dumped {0} refs for {1}'.format(len(ref), cid))


def build_summary_targets_annually(year, single_file, tokenize, manual=False):
    if config.data_mode == 'mixed':
        cc_ids = tools.get_mixed_cc_ids_annually(year, model_mode='test')
        out_dp = join(path_parser.data_summary_targets, 'mixed')
    elif config.data_mode == 'sep':
        cc_ids = tools.get_cc_ids(year, model_mode='test')
        if tokenize:
            out_dir = '{}_tokenized'.format(year)
        elif manual:
            out_dir = '{}_manual'.format(year)
        else:
            out_dir = year
        out_dp = join(path_parser.data_summary_targets, out_dir)
    else:
        raise ValueError('Invalid data_mode: {}'.format(config.data_mode))

    if manual:
        in_dir = '{}_manual'.format(year)
    else:
        in_dir = year
    ref_dp = join(path_parser.data_summary_refs, in_dir)
    cid2ref_fps = retrieve_refs_fps_with_cid(cc_ids, ref_dp)

    if single_file:
        build_summary_targets_single_file(cid2ref_fps, out_dp)
    else:
        build_summary_targets(cid2ref_fps, out_dp, tokenize=tokenize)


def build_summary_targets_end2end(single_file, tokenize, manual=False):
    for year in config.years:
        build_summary_targets_annually(year, single_file, tokenize, manual)


if __name__ == '__main__':
    # build_summary_targets(year='2005')
    build_summary_targets_annually(year='2007', single_file=False, tokenize=False, manual=True)
