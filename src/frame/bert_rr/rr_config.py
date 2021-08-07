import utils.config_loader as config
import ir.ir_config as ir_config

# set following macro vars
# For DUC: config.QUERY
# For TD-QFS: pred@grsum-tdqfs-0.6_cos-0_wan-nw_250@masked-ratio-reveal_1.0
QUERY_TYPE = config.QUERY

NO_QUERY = True if not QUERY_TYPE else False
if config.rr_config_id in [14, 15, 16, 17, 18, 19]:
    WITH_SUB = True
else:
    WITH_SUB = False

# IR configs: the method should be consistent
IR_MODEL_NAME = ir_config.IR_MODEL_NAME_TF  # for building sid2sent for contextual QA models

# IR_RECORDS_DIR_NAME: sentence lookup, IR_MODEL_NAME_TF (full), IR_RECORDS_DIR_NAME_TF (retrieved)
IR_RECORDS_DIR_NAME = ir_config.IR_MODEL_NAME_TF

if config.meta_model_name == 'bert_rr':
    BERT_TYPE = f'{config.rr_config_id}_config-{config.rr_iter}_iter'
    RR_MODEL_NAME_BERT = f'rr-{BERT_TYPE}-{QUERY_TYPE}-{IR_RECORDS_DIR_NAME}'
elif config.meta_model_name == 'bert_qa':  # for ablation study: build unilm in from qa_tdqfs and eval unilm
    BERT_TYPE = 'bert'
    RR_MODEL_NAME_BERT = f'qa-{BERT_TYPE}-narr-{IR_RECORDS_DIR_NAME}'
else:
    raise ValueError(f'Invalid mode_name: {config.meta_model_name}')

RELEVANCE_SCORE_DIR_NAME = RR_MODEL_NAME_BERT

#### for Multi-News
QUERY_TYPE_MN = 'masked_summ'
if QUERY_TYPE_MN in (config.QUERY, config.NARR, config.TITLE):  # omit typical query type for DUC
    RR_MODEL_NAME_BERT_MN = f'rr-{BERT_TYPE}-mn'
else:
    RR_MODEL_NAME_BERT_MN = f'rr-{BERT_TYPE}-{QUERY_TYPE_MN}-mn'
RELEVANCE_SCORE_DIR_NAME_MN = RR_MODEL_NAME_BERT_MN

# filter config
FILTER = 'topK'  # topK, conf

CONF_THRESHOLD_RR = 0.4  # 0.95, 0.75
COMP_RATE_RR = 0.85
# 40~150, 
# QA: 90: sentence, 110: passage
TOP_NUM_RR = 150

if FILTER == 'conf':
    FILTER_VAR = CONF_THRESHOLD_RR
elif FILTER == 'comp':
    FILTER_VAR = COMP_RATE_RR
elif FILTER == 'topK':
    FILTER_VAR = TOP_NUM_RR
else:
    raise ValueError(f'Invalid FILTER: {FILTER}')

RR_RANK_DIR_NAME_BERT = RR_MODEL_NAME_BERT
RR_RECORD_DIR_NAME_PATTERN = 'rr_records-{}-{}_qa_{}'  # model name, conf

RR_MIN_NS = None  # None (turn off), 50
if RR_MIN_NS:
    RR_RECORD_DIR_NAME_PATTERN += f'-{RR_MIN_NS}_rr_min_ns'

RR_INTERPOLATION = None # None (turn off), 0.5
RR_INTERPOLATION_NORM = True
if RR_INTERPOLATION:
    RR_RECORD_DIR_NAME_PATTERN += f'-{RR_INTERPOLATION}_rr_interpolation'
    if RR_INTERPOLATION_NORM:
        RR_RECORD_DIR_NAME_PATTERN += '-norm'

RR_RECORD_DIR_NAME_BERT = RR_RECORD_DIR_NAME_PATTERN.format(RR_MODEL_NAME_BERT, FILTER_VAR, FILTER)
RR_RECORD_DIR_NAME_BERT_MN = RR_RECORD_DIR_NAME_PATTERN.format(RR_MODEL_NAME_BERT_MN, FILTER_VAR, FILTER)

RR_TUNE_DIR_NAME_BERT = f'rr_tune-{RR_MODEL_NAME_BERT}'
RR_FINER_TUNE_DIR_NAME_BERT = f'rr_finer_tune-{RR_MODEL_NAME_BERT}'

RR_TUNE_DIR_NAME_BERT_MN = f'rr_tune-{RR_MODEL_NAME_BERT_MN}'
RR_FINER_TUNE_DIR_NAME_BERT_MN = f'rr_finer_tune-{RR_MODEL_NAME_BERT_MN}'

"""
    Below are configs for unilm input; copied from marge_config.py
        - UNILM_IN_FILE_NAME
        - POSITIONAL
        - PREPEND_LEN
"""
USE_TEXT=False
if USE_TEXT:  # can sentences selected by rr or centrality
    # centrality-hard_bias-0.85_damp-rr_records-rr-0_config-45000_iter-mn-90_qa_topK-0.6_cos-4_wan
    # centrality-hard_bias-0.85_damp-rr_records-rr-34_config-25000_iter-query-ir-dial-tf-2006-150_qa_topK-0.6_cos-10_wan
    centrality_dir_name = 'centrality-hard_bias-0.85_damp-rr_records-rr-34_config-25000_iter-query-ir-dial-tf-2006-150_qa_topK-0.6_cos-10_wan'
    # rr-34_config-25000_iter-query-ir-dial-tf-2007-0.6_cos
    # rr-39_config-26000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-0.6_cos-nw_250
    rr_select_dir_name = 'rr-39_config-26000_iter-add_left_mask_right_dot-ir-dial-tf-tdqfs-0.6_cos-nw_250'
    TEXT_DIR_NAME = rr_select_dir_name
    UNILM_IN_FILE_NAME = f'unilm_in-{TEXT_DIR_NAME}'
else:
    UNILM_IN_FILE_NAME = f'unilm_in-{RR_MODEL_NAME_BERT}-top{FILTER_VAR}'

POSITIONAL=None  # global, local, None
if POSITIONAL:
    RR_RANK_DIR_NAME_BERT += '-w_pos'
    UNILM_IN_FILE_NAME += f'-{POSITIONAL}_pos'

PREPEND_LEN=True
if PREPEND_LEN:
    UNILM_IN_FILE_NAME += '-prepend_len'

PREPEND_QUERY='masked'  # 'raw', 'masked', None, add_left_mask_right_dot
if PREPEND_QUERY:
    UNILM_IN_FILE_NAME += f'-prepend_{PREPEND_QUERY}_q'

UNILM_IN_FILE_NAME += '.json'

"""
    Below are configs for unilm eval; copied from marge_config.py
        - UNILM_DECODE_FILE_NAME
        - UNILM_OUT_DIR_NAME

"""

from pathlib import Path
UNILM_MODEL_ID2CKPT = {
    7: 3000,
    17: 2500,
}
UNILM_MODEL_ID = 17 # 7 (MN), 17 (MCD)
UNILM_CKPT = UNILM_MODEL_ID2CKPT[UNILM_MODEL_ID]

UNILM_MODEL_ROOT = Path('~/unilm/model')
if USE_TEXT:
    UNILM_DECODE_FILE_NAME = f'ckpt-{UNILM_CKPT}.{TEXT_DIR_NAME}'
    UNILM_OUT_DIR_NAME = f'unilm_{UNILM_MODEL_ID}_{UNILM_CKPT}-{TEXT_DIR_NAME}'
else:
    UNILM_DECODE_FILE_NAME = f'ckpt-{UNILM_CKPT}.{RR_MODEL_NAME_BERT}-top{FILTER_VAR}'
    UNILM_OUT_DIR_NAME = f'unilm_{UNILM_MODEL_ID}_{UNILM_CKPT}-{RR_MODEL_NAME_BERT}-top{FILTER_VAR}'

if POSITIONAL:
    UNILM_DECODE_FILE_NAME += f'-{POSITIONAL}_pos'
    UNILM_OUT_DIR_NAME += f'-{POSITIONAL}_pos'

if PREPEND_LEN:
    UNILM_DECODE_FILE_NAME += '-prepend_len'
    UNILM_OUT_DIR_NAME += '-prepend_len'

if PREPEND_QUERY:
    UNILM_DECODE_FILE_NAME += f'-prepend_{PREPEND_QUERY}_q'
    UNILM_OUT_DIR_NAME += f'-prepend_{PREPEND_QUERY}_q'

DECODE_AFFIX = ''
if DECODE_AFFIX:
    UNILM_DECODE_FILE_NAME += f'-{DECODE_AFFIX}'
    UNILM_OUT_DIR_NAME += f'-{DECODE_AFFIX}'

UNILM_DECODE_FILE_PATH = UNILM_MODEL_ROOT / f'unilm_{UNILM_MODEL_ID}' / UNILM_DECODE_FILE_NAME


def override_glob_vars(unilm_model_id, unilm_ckpt):
    global UNILM_DECODE_FILE_NAME, UNILM_OUT_DIR_NAME, UNILM_DECODE_FILE_NAME, UNILM_OUT_DIR_NAME, UNILM_DECODE_FILE_PATH
    if USE_TEXT:
        UNILM_DECODE_FILE_NAME = f'ckpt-{unilm_ckpt}.{TEXT_DIR_NAME}'
        UNILM_OUT_DIR_NAME = f'unilm_{unilm_model_id}_{unilm_ckpt}-{TEXT_DIR_NAME}'
    else:
        UNILM_DECODE_FILE_NAME = f'ckpt-{unilm_ckpt}.{RR_MODEL_NAME_BERT}-top{FILTER_VAR}'
        UNILM_OUT_DIR_NAME = f'unilm_{unilm_model_id}_{unilm_ckpt}-{RR_MODEL_NAME_BERT}-top{FILTER_VAR}'

    if POSITIONAL:
        UNILM_DECODE_FILE_NAME += f'-{POSITIONAL}_pos'
        UNILM_OUT_DIR_NAME += f'-{POSITIONAL}_pos'

    if PREPEND_LEN:
        UNILM_DECODE_FILE_NAME += '-prepend_len'
        UNILM_OUT_DIR_NAME += '-prepend_len'

    if PREPEND_QUERY:
        UNILM_DECODE_FILE_NAME += f'-prepend_{PREPEND_QUERY}_q'
        UNILM_OUT_DIR_NAME += f'-prepend_{PREPEND_QUERY}_q'

    if DECODE_AFFIX:
        UNILM_DECODE_FILE_NAME += f'-{DECODE_AFFIX}'
        UNILM_OUT_DIR_NAME += f'-{DECODE_AFFIX}'

    UNILM_DECODE_FILE_PATH = UNILM_MODEL_ROOT / f'unilm_{unilm_model_id}' / UNILM_DECODE_FILE_NAME
    