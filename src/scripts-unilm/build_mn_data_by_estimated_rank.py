import io
import os
from os.path import dirname, abspath, join, exists
from pathlib import Path
from tqdm import tqdm
import json
import nltk


DATASET_VAR = 'train'  # specify this 

SHIFTSUM_ROOT = Path('~/margesum/data/multinews')
UNILM_ROOT = Path('~/unilm/data')

ESTIMATED_ROUGE_NAME = 'rouge_estimated-rr-34_config-25000_iter-masked_summ-mn-marge_cluster-ratio-reveal_0.0'

FINAL_DATA_DIR_NAME = ESTIMATED_ROUGE_NAME
TGT_MIN_WORDS = None
if TGT_MIN_WORDS:
    FINAL_DATA_DIR_NAME += f'-{TGT_MIN_WORDS}'

PREPEND_LEN = True
if PREPEND_LEN:
    FINAL_DATA_DIR_NAME += '_prepend_len'

SWAP_PROB = 0.0
if SWAP_PROB > 0.0:
    FINAL_DATA_DIR_NAME += f'_{SWAP_PROB}_swap'

FINAL_DATA_DIR = UNILM_ROOT / FINAL_DATA_DIR_NAME


def get_cid2summary():
    masked_summary_fp = SHIFTSUM_ROOT / 'masked_mn_summary' / f'{DATASET_VAR}-sample-max_reveal_1.0.json'
    cid = 0
    cid2summary = {}
    with open(masked_summary_fp) as masked_summary_f:
        for line in masked_summary_f:
            json_obj = json.loads(line)
            cid2summary[cid] = json_obj['original_summary']
            cid += 1
    return cid2summary


def _get_cid(json_obj):
    return int(json_obj['sid'].split('_')[0])


def _swap_sentence_objs(sentence_objs, metric, swap_prob):
    def _swap(i, j):
        temp = sentence_objs[j]
        sentence_objs[j] = sentence_objs[i]
        sentence_objs[i] = temp
        
    status = [0] * len(sentence_objs)

    for ii, so in enumerate(sentence_objs[:-1]):
        if status[ii] == 1:  # has bee swapped
            continue
            
        do_swap = np.random.choice([0, 1], p=np.array([1.0-swap_prob, swap_prob]))
        if not do_swap:
            continue
        
        candidates = sentence_objs[ii+1:]
        for relative_pos, _so in enumerate(candidates):
            score = 1.0 / (math.abs(so[metric]-_so[metric]) + 1e-7)
            if status[jj] == 1:
                score = 0.0
            scores.append(score)
        nom = sum(scores)

        prob_dist = np.array([sc/nom for sc in scores])
        indices = ii + 1 + np.arange(len(prob_dist))
        jj = np.random.choice(indices, p=prob_dist)  # abs_pos
        _swap(ii, jj)


def get_len_token(tgt_len):
    if tgt_len < 100:
        tgt_len = 85
    elif tgt_len >= 400:
        tgt_len = 400
    else:
        for start in range(100, 400, 15):
            if start <= tgt_len < start+15:
                tgt_len = start
                break
    
    assert (tgt_len-100)%15==0 or tgt_len==85, f'{tgt_len} is not right'
    return f'[unused{tgt_len}]'


def unit_test_get_len_token():
    tgt_lens = [99, 100, 101, 201, 250, 399, 400, 401]
    for tl in tgt_lens:
        token = get_len_token(tl)
        print(f'{tl}\t{token}')


def to_save(tgt_len):
    to_save = True
    if TGT_MIN_WORDS and tgt_len < TGT_MIN_WORDS:
        to_save = False
    
    return to_save


def build():
    rouge_fp = SHIFTSUM_ROOT / ESTIMATED_ROUGE_NAME / f'{DATASET_VAR}.json'
    cid2summary = get_cid2summary()

    if not exists(FINAL_DATA_DIR):
        os.mkdir(FINAL_DATA_DIR)

    dump_fp = FINAL_DATA_DIR / f'{DATASET_VAR}.json'
    if exists(dump_fp):
        raise ValueError(f'dump_fp exists: {dump_fp}')

    cid = 0
    sentence_objs = []
    with open(dump_fp, 'a') as dump_f:
        with open(rouge_fp) as rouge_f:
            for line in rouge_f:
                line = line.strip('\n')
                if not line:
                    continue
                json_obj = json.loads(line)
                _cid =  _get_cid(json_obj)
                if _cid != cid:
                    if SWAP_PROB > 0.0:
                        _swap_sentence_objs(sentence_objs, metric=METRIC, swap_prob=SWAP_PROB)

                    if cid % 1000 == 0:
                        print(f'cid: {cid}, #Sentences: {len(sentence_objs)}')
                    
                    tgt = cid2summary[cid]
                    tgt_words = nltk.tokenize.word_tokenize(tgt)
                    tgt_len = len(tgt_words)

                    if to_save(tgt_len):
                        sentences = [so['sentence'].replace('NEWLINE_CHAR', '').strip()
                            for so in sentence_objs]
                        src = ' '.join(sentences)

                        if PREPEND_LEN:
                            src = get_len_token(tgt_len) + ' ' + src
                        
                        dump_obj = {
                            "sentences": sentence_objs,
                            "src": src,
                            "tgt": tgt,
                        }
                        json_str = json.dumps(dump_obj)
                        dump_f.write(f'{json_str}\n')

                    sentence_objs = []

                so = {
                    'id': json_obj['sid'],
                    'sentence': json_obj['sent'],
                    'estimated_rouge': float(json_obj['estimated_rouge']),
                }
                sentence_objs.append(so)
                cid = _cid
    
    print(f'Sucessfully dump {DATASET_VAR} set to: {dump_fp}!')


if __name__ == "__main__":
    build()
