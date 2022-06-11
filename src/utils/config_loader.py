import logging
import logging.config
import yaml
from io import open
import os
from os.path import join, dirname, abspath
import socket
import warnings
import sys
from pytorch_transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, BertForQuestionAnswering
import torch
from pathlib import Path

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def deprecated(func):
    """
        This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.
    """

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


"""
    Most of the code in this file is borrowed from QuerySum codebase.

"""

class PathParser:
    def __init__(self, path_type):
        self.remote_root = Path('/disk/nfs/ostrom')

        if path_type == 'local':
            self.proj_root = Path('~/margesum')
        elif path_type == 'afs':
            self.proj_root = self.remote_root / 'margesum'
        else:
            raise ValueError(f'Invalid path_type: {path_type}')
        print(f'Set proj_root to: {self.proj_root}')
        
        self.performances = self.proj_root / 'performances'
        self.log = self.proj_root / 'log'
        self.data = self.proj_root / 'data'
            
        # multinews
        self.multinews_raw = self.data / 'multinews' / 'raw'
        self.top_mn = self.data / 'top_mn'

        # cnndm
        self.cnndm_raw = self.data / 'cnndm' / 'raw'
        self.cnndm_url = self.cnndm_raw / 'url_lists'
        self.cnndm_raw_cnn_story = self.cnndm_raw / 'cnn' / 'stories'
        self.cnndm_raw_dm_story = self.cnndm_raw / 'dailymail' / 'stories'
        
        # duc
        self.duc_cluster = self.data / 'duc_cluster'
        self.data_docs = self.data / 'duc_cluster'

        # tdqfs
        self.data_tdqfs = self.data / 'tdqfs'
        self.data_tdqfs_sentences = self.data_tdqfs / 'sentences'
        self.data_tdqfs_queries = self.data_tdqfs / 'query_info.txt'
        self.data_tdqfs_summary_targets = self.data_tdqfs / 'summary_targets'

        self.raw_query = self.data / 'raw_query'
        self.parsed_query = self.data / 'parsed_query'
        self.masked_query = self.data / 'masked_query'

        self.data_summary_targets = self.data / 'duc_summary'
        self.data_mn_summary_targets = self.data / 'multinews_rr' / 'test_mn_summary'

        # for UniLM
        self.unilm_in = self.proj_root / 'unilm_in'
        self.unilm_out = self.proj_root / 'unilm_out'
        
        # set res
        self.res = self.proj_root / 'res'

        self.model_save = self.proj_root / 'model'

        self.pred = self.proj_root / 'pred'

        self.summary_rank = self.proj_root / 'rank'
        self.summary_text = self.proj_root / 'text'
        self.graph = self.proj_root / 'graph'

        self.graph_rel_scores = self.graph / 'rel_scores'  # for dumping relevance scores
        self.graph_token_logits = self.graph / 'token_logits'  # for dumping relevance scores

        self.rouge = self.proj_root / 'rouge'
        self.tune = self.proj_root / 'tune'
        self.cont_sel = self.proj_root / 'cont_sel'
        self.mturk = self.proj_root / 'mturk'

        self.afs_rouge_dir = self.remote_root / 'ROUGE-1.5.5' / 'data'
        self.local_rouge_dir = '~/pyrouge/RELEASE-1.5.5/data'


config_root = Path(os.path.dirname(os.path.dirname(__file__))) / 'config'

# meta
config_meta_fp = config_root / 'config_meta.yml'
config_meta = yaml.load(open(config_meta_fp, 'r', encoding='utf-8'))
path_type = config_meta['path_type']
mode = config_meta['mode']

path_parser = PathParser(path_type=path_type)

# model
meta_model_name = config_meta['model_name']

# join_query_para
# True: use one enc for the concatenation of query and para;
# False: separate encoders for query and para.
config_model_fn = f'config_model_{meta_model_name}.yml'
config_model_fp = config_root / config_model_fn
config_model = yaml.load(open(config_model_fp, 'r'))

test_year = config_meta['test_year']
model_name = config_model['model_name']

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = join(path_parser.log, '{0}.log'.format(model_name))
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s", "%m/%d/%Y %H:%M:%S")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.info(f'model name: {model_name}')

NARR = 'narr'
TITLE = 'title'
QUERY = 'query'
NONE = 'None'
HEADLINE = 'headline'
NEG_LABEL_POS = '0'
NEG_LABEL_NARR = '1'
NEG_LABEL_HEADLINE = '2'
SEP = '_'
NEG_SEP = ';'

query_types = (NARR, HEADLINE)
years = ['2005', '2006', '2007']
baselines = ['lead', 'graph_tfidf', 'graph_tfidf-whole_query']


def load_bert_qa():
    # root = path_parser.model_save / 'saved.qa_2'
    # print('Load PyTorch model from {}'.format(root))
    # if not torch.cuda.is_available():  # cpu
    #     state = torch.load(root, map_location='cpu')
    # else:
    #     state = torch.load(root)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # return state['epoch'], state['model'], state['tokenizer'], state['scores']
    return None, None, bert_tokenizer, None


rr_config_id2iter = {
    'multinews': 25000,
    'cnndm': 26000,
}

rr_config_id = 'multinews'  # multinews, cnndm
rr_iter = rr_config_id2iter[rr_config_id]

def load_bert_rr():
    root = path_parser.model_save / f'marge.{rr_config_id}' / f'checkpoint-{rr_iter}'
    rr_config_fp = root / 'config.json'
    rr_model_fp = root / 'pytorch_model.bin'
    
    config = BertConfig.from_json_file(rr_config_fp)
    bert_model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=rr_model_fp, config=config)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True,
        do_basic_tokenize=True, additional_special_tokens=['[SLOT]', '[SUBQUERY]'])
    
    return bert_model, bert_tokenizer


preload_model_tokenizer = config_meta['preload_model_tokenizer']
if preload_model_tokenizer:
    if meta_model_name == 'bert_rr' and config_model['fine_tune'] == 'rr':
        logger.info('building BERT model and tokenizer: {}'.format(config_model['fine_tune']))
        bert_model, bert_tokenizer = load_bert_rr()
    elif meta_model_name == 'bert_qa' and config_model['fine_tune'] == 'qa':
        logger.info('building BERT model and tokenizer: {}'.format(config_model['fine_tune']))
        _, bert_model, bert_tokenizer, _ = load_bert_qa()
    else:
        logger.info('building BERT tokenizer')
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


if mode == 'rank_sent':
    config_model['d_batch'] = 50
