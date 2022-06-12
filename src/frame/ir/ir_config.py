import utils.config_loader as config
from utils.config_loader import config_meta

if config.grain == 'sent':
    IR_META_NAME = 'ir'
else:
    IR_META_NAME = 'ir-{}'.format(config.grain)  # e.g., ir-passage

QUERY_TYPE = None  # config.NARR, config.TITLE, None, REF
if QUERY_TYPE:
    CONCAT_TITLE_NARR = False
    IR_META_NAME = '{}-{}'.format(IR_META_NAME, QUERY_TYPE)
else:
    CONCAT_TITLE_NARR = True

REMOVE_DIALOG = False
if not REMOVE_DIALOG:
    IR_META_NAME = '{}-{}'.format(IR_META_NAME, 'dial')

# test_year = config.test_year
test_year = config_meta['test_year']  # may contain robustness information (e.g., affix with 10/30/50)
IR_MODEL_NAME_TFIDF = '{}-tfidf-{}'.format(IR_META_NAME, test_year)
IR_MODEL_NAME_BM25 = '{}-bm25-{}'.format(IR_META_NAME, test_year)
IR_MODEL_NAME_TF = '{}-tf-{}'.format(IR_META_NAME, test_year)
IR_MODEL_NAME_TF_DOT = '{}-tf_dot-{}'.format(IR_META_NAME, test_year)
IR_MODEL_NAME_COUNT = '{}-count-{}'.format(IR_META_NAME, test_year)
IR_MODEL_NAME_WEIGHTED_COUNT = '{}-weighted_count-{}'.format(IR_META_NAME, test_year)
IR_MODEL_NAME_BHA = '{}-bha-{}'.format(IR_META_NAME, test_year)

DEDUPLICATE = False

CONF_THRESHOLD_IR = 0.75  # 0.75, 0.9, 0.95, 0.98
COMP_RATE_IR = 0.25
TOP_NUM_IR = 90

IR_MIN_NS = None  # None (turn off), 100, 50

FILTER = 'conf'  # conf, comp, topK (only used in ablation study)

if FILTER == 'conf':
    FILTER_VAR = CONF_THRESHOLD_IR
elif FILTER == 'comp':
    FILTER_VAR = COMP_RATE_IR
elif FILTER == 'topK':
    FILTER_VAR = TOP_NUM_IR
else:
    raise ValueError('Invalid FILTER: {}'.format(FILTER))

IR_RECORDS_DIR_NAME_PATTERN = 'ir_records-{}-{}_ir_{}'

if DEDUPLICATE:
    IR_RECORDS_DIR_NAME_PATTERN += '-dedup'

if IR_MIN_NS:
    IR_RECORDS_DIR_NAME_PATTERN += '-{}_ir_min_ns'.format(IR_MIN_NS)

IR_RECORDS_DIR_NAME_TFIDF = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_TFIDF, FILTER_VAR, FILTER)
IR_RECORDS_DIR_NAME_TF = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_TF, FILTER_VAR, FILTER)
IR_RECORDS_DIR_NAME_TF_DOT = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_TF_DOT, FILTER_VAR, FILTER)
IR_RECORDS_DIR_NAME_COUNT = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_COUNT, FILTER_VAR, FILTER)
IR_RECORDS_DIR_NAME_WEIGHTED_COUNT = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_WEIGHTED_COUNT, FILTER_VAR, FILTER)
IR_RECORDS_DIR_NAME_BM25 = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_BM25, FILTER_VAR, FILTER)
IR_RECORDS_DIR_NAME_BHA = IR_RECORDS_DIR_NAME_PATTERN.format(IR_MODEL_NAME_BHA, FILTER_VAR, FILTER)

IR_TUNE_DIR_NAME_PATTERN = 'ir_tune-{}'
IR_TUNE_DIR_NAME_TFIDF = IR_TUNE_DIR_NAME_PATTERN.format(IR_MODEL_NAME_TFIDF)
IR_TUNE_DIR_NAME_TF = IR_TUNE_DIR_NAME_PATTERN.format(IR_MODEL_NAME_TF)
IR_TUNE_DIR_NAME_TF_DOT = IR_TUNE_DIR_NAME_PATTERN.format(IR_MODEL_NAME_TF_DOT)
IR_TUNE_DIR_NAME_COUNT = IR_TUNE_DIR_NAME_PATTERN.format(IR_MODEL_NAME_COUNT)
IR_TUNE_DIR_NAME_WEIGHTED_COUNT = IR_TUNE_DIR_NAME_PATTERN.format(IR_MODEL_NAME_WEIGHTED_COUNT)
IR_TUNE_DIR_NAME_BM25 = IR_TUNE_DIR_NAME_PATTERN.format(IR_MODEL_NAME_BM25)
IR_TUNE_DIR_NAME_BHA = IR_TUNE_DIR_NAME_PATTERN.format(IR_MODEL_NAME_BHA)

"""
    Below are configs for unilm input; copied from marge_config.py
        - UNILM_IN_FILE_NAME
        - POSITIONAL
        - PREPEND_LEN
        - MULTI_PASS

"""
USE_CENTRALITY=False
if USE_CENTRALITY:
    CENTRALITY_DIR_NAME = \
        'centrality-hard_bias-0.85_damp-rr_records-rr-0_config-45000_iter-mn-90_qa_topK-0.6_cos-4_wan'
    UNILM_IN_FILE_NAME = f'unilm_in-{CENTRALITY_DIR_NAME}'
else:
    UNILM_IN_FILE_NAME = f'unilm_in-{IR_MODEL_NAME_TF}-conf{FILTER_VAR}'

POSITIONAL=None  # global, local, None
if POSITIONAL:
    UNILM_IN_FILE_NAME += f'-{POSITIONAL}_pos'

PREPEND_LEN=True
if PREPEND_LEN:
    UNILM_IN_FILE_NAME += '-prepend_len'

MULTI_PASS=False
if MULTI_PASS:
    UNILM_IN_FILE_NAME += '-multipass'

UNILM_IN_FILE_NAME += '.json'

"""
    Below are configs for unilm eval; copied from rr_config.py
        - UNILM_DECODE_FILE_NAME
        - UNILM_OUT_DIR_NAME

"""

from pathlib import Path
UNILM_MODEL_ID2CKPT = {
    2: 33000,
    4: 4500,
    5: 3000,
    6: 10500,
    7: 10500,
    8: 10500,
    9: 15000,
    10: 10500,
}
UNILM_MODEL_ID = 7
UNILM_CKPT = UNILM_MODEL_ID2CKPT[UNILM_MODEL_ID]

UNILM_MODEL_ROOT = Path('/home/s1617290/unilm/model')
if USE_CENTRALITY:
    UNILM_DECODE_FILE_NAME = f'ckpt-{UNILM_CKPT}.{CENTRALITY_DIR_NAME}'
    UNILM_OUT_DIR_NAME = f'unilm_{UNILM_MODEL_ID}_{UNILM_CKPT}-{CENTRALITY_DIR_NAME}'
else:
    UNILM_DECODE_FILE_NAME = f'ckpt-{UNILM_CKPT}.{IR_MODEL_NAME_TF}-conf{FILTER_VAR}'
    UNILM_OUT_DIR_NAME = f'unilm_{UNILM_MODEL_ID}_{UNILM_CKPT}-{IR_MODEL_NAME_TF}-conf{FILTER_VAR}'

if POSITIONAL:
    UNILM_DECODE_FILE_NAME += f'-{POSITIONAL}_pos'
    UNILM_OUT_DIR_NAME += f'-{POSITIONAL}_pos'

if PREPEND_LEN:
    UNILM_DECODE_FILE_NAME += '-prepend_len'
    UNILM_OUT_DIR_NAME += '-prepend_len'

if MULTI_PASS:
    UNILM_DECODE_FILE_NAME += '-multipass'
    UNILM_OUT_DIR_NAME += '-multipass'

DECODE_AFFIX = ''  # 300-PP, 400, PP-step_0.1
if DECODE_AFFIX:
    UNILM_DECODE_FILE_NAME += f'-{DECODE_AFFIX}'
    UNILM_OUT_DIR_NAME += f'-{DECODE_AFFIX}'

UNILM_DECODE_FILE_PATH = UNILM_MODEL_ROOT / f'unilm_{UNILM_MODEL_ID}' / UNILM_DECODE_FILE_NAME