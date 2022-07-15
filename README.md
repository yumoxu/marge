# marge

This repository releases the code for Generating Query Focused Summaries from Query-Free Resources. 

Please cite the following paper [[bib]](https://aclanthology.org/2021.acl-long.475.bib) if you use this code,

Xu, Yumo, and Mirella Lapata. "Generating Query Focused Summaries from Query-Free Resources." In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 6096–6109. 2021.

> The availability of large-scale datasets has driven the development of neural models that create generic summaries from single or multiple documents. In this work we consider query focused summarization (QFS), a task for which training data in the form of queries, documents, and summaries is not readily available. We propose to decompose QFS into (1) query modeling (i.e., finding supportive evidence within a set of documents for a query) and (2) conditional language modeling (i.e., summary generation). We introduce MaRGE, a Masked ROUGE Regression framework for evidence estimation and ranking which relies on a unified representation for summaries and queries, so that summaries in generic data can be converted into proxy queries for learning a query model. Experiments across QFS benchmarks and query types show that our model achieves state-of-the-art performance despite learning from weak supervision.

Should you have any query please contact me at yumo.xu@ed.ac.uk.


# Preliminary setup

## Project structure
```bash
marge
└───requirements.txt
└───README.md
└───log        # logging files
└───run        # scripts for MaRGE training
└───src        # source files
└───data       # generic data for training; qfs data for test/dev
└───graph      # graph components for query expansion
└───model      # MaRGE models for inference
└───rank       # ranking results
└───text       # summarization results
└───unilm_in   # input files to UniLM
└───unilm_out  # output files from UniLM
```

After cloning this project, use the following command to initialize the structure:
```bash
mkdir log data graph model rank text unilm_in unilm_out
touch log/BertRR.log
```

## Creating environment
```bash
cd ..
virtualenv -p python3.6 marge
cd marge
. bin/activate
pip install -r requirements.txt
```
You need to install apex:
```bash
cd ..
git clone https://www.github.com/nvidia/apex
cd apex
python3 setup.py install
```

Also, you need to setup ROUGE evaluation if you have not yet done it. Please refer to [this](https://github.com/bheinzerling/pyrouge) repository. After finishing the setup, specify the ROUGE path in `frame/utils/config_loader.py` as an attribute of `PathParser`:
```python
self.rouge_dir = '~/ROUGE-1.5.5/data'  # specify your ROUGE dir
```

## Preparing benchmark data
Since we are not allowed to distribute **DUC** clusters and summaries, you can request DUC 2005-2007 from [NIST](https://www-nlpir.nist.gov/projects/duc/data.html). After acquiring the data, gather each year's clusters and summaries under `data/duc_cluster` and `data/duc_summary`, respectively. 
For instance, DUC 2006's clusters and  summaries should be found under `data/duc_cluster/2006/` and `data/duc_summary/2006/`, respectively. 
For DUC queries: you don't have to prepare queries by yourself; we have put 3 `json` files for DUC 2005-2007 under `data/masked_query`, which contain a raw query and a masked query for each cluster. Queries will be fetched from these files at test time.

**TD-QFS** data can be downloaded from [here](https://talbaumel.github.io/TD-QFS/files/TD-QFS.zip).
You can also use the processed version [here](https://drive.google.com/file/d/1fPZFRAfaojJNEeyk7cKEMlVti1UhIfFs/view?usp=sharing).

After data preparation, you should have the following directory structure with the right files under each folder:

```bash
marge
└───data
│   └───duc_clusters   # DUC clusters 
│   └───duc_summaries  # DUC reference summaries 
│   └───masked_query   # DUC queries (raw and masked)
│   └───tdqfs          # TD-QFS clusters, queries and reference summaries
```
## Pretrained models
Please use [this](https://drive.google.com/drive/folders/10vOlQaJYplztSdBnNe51MqHNvHuMmymv?usp=sharing) link to access the checkpoints, including the best performing pipeline of:
1. `marge.multinews`: an evidence ranker trained on Multi-News, and
2. `margesum.cnndm`: a summary generator trained on CNN/DM. 

Put the zip files under `marge/model` and then unzip them.

# MaRGE: query modeling
## Preparing training data
The first step is to download raw Multi-News and CNN/DM data. Put them under `data/{dataset}/raw` where `dataset` can `multinews` or `cnndm`. 
- For MultiNews, prepare `.src.cleaned` files ([download link](https://drive.google.com/drive/folders/1jwBzXBVv8sfnFrlzPnSUBHEEAbpIUnFq)) and `.tgt` files ([download link](https://drive.google.com/drive/folders/1uDarzpu2HFc-vjXNJCRv2NIHzakpSGOw)).
- For CNN/DM, prepare `url_lists`, `cnn/stories` and `dailymail/stories` ([download link](https://github.com/abisee/cnn-dailymail)). 

Source files for building training data are under `src/sripts`. For each dataset (Multi-News or CNN/DM), there are three  steps  create MaRGE training data. 

A training sample for MaRGE can be represented as {sentence, masked summary}->ROUGE(sentence, summary). So we need to get the ROUGE scores for all sentences (step 1) and creating masked summaries (step 2). Then we put them together (step 3).

1. Calculate ROUGE scores for all sentences: 

  ```python
  python src/sripts/dump_sentence_rouge_mp.py
  ```

2. Build masked summaries:

  ```python
  python src/sripts/mask_summary_with_ratio.py
  ```

3. Build train/val/test datasets:

  ```python
  python src/sripts/build_marge_dataset_mn.py
  ```

In our experiments, MaRGE trained on data from **Multi-News** yielded the best performance in query modeling. 
If you want to build training data from CNN/DM:
1. Use the function `gathered_mp_dump_sentence_cnndm()` in the first step (otherwise, use the function `gathered_mp_dump_sentence_mn()` )
2. Set `dataset='cnndm'` in the second step (otherwise, `dataset='mn'`)
3. Use `build_marge_dataset_cnndm.py` instead for the last step

**Question**: *What should I do if I want to use MaRGE on my own dataset?* 

**Answer**: start with building a customized data parser for your dataset at hand. 
The two existing parsers included in this project, `mn_parser.py` and `cnndm_parser.py`, can be found under `src/data`. 
You can write a new, structurally similar parser class, which can then be imported to the scripts introduced in this section to reuse the pipeline. For instance, in `dump_sentence_rouge_mp.py`, [this](https://github.com/yumoxu/marge/blob/be003cf9db8caed431b4d6717c903343e6713cda/src/scripts/dump_sentence_rouge_mp.py#L20) line now imports the parser for `MultiNewsParser`, but can be altered to any other parser added under `src/data` for your customized data.


## Model training 

Depending on the training data (CNN/DM or Multi-News), you can run one of the scripts under `run/`. 
Configs specified in these two files are used in our experiments, but feel free to change them for further experimentation.

## Inference and evaluation

Use `src/frame/rr/main.py` for DUC evaluation and `src/frame/rr/main_tdqfs.py` for TD-QFS evalaution. We will take DUC evaluation for example.

In `src/frame/rr/main.py`, run the following methods in order (or at once):
```python
init()
dump_rel_scores()  # inference with MaRGE
rel_scores2rank()  # turn sentence scores to sentence rank
rr_rank2records()  # take top sentences
```
To evaluate evidence rank, in `src/frame/rr/main.py`, run:

```python
select_e2e()
```

# MaRGESum: summary generation

## System output release
You can access the system outputs for MaRGESum-CD and MaRGESum-MN on DUC 2006-07 and TDQFS via [this](https://drive.google.com/drive/folders/1YN8bg8j5aQ0J1YUeRYI_6Lmhsgpf6UH8?usp=sharing) link. 

## Prepare training data from Multi-News
To train a controllable generator, we make the following three changes to the input from Multi-News (and CNN/DM):
1. Re-order input sentences according to their ROUGE scores, so the top ones will be biased over:
```Python
python scripts/selector_for_train.py
```
2. Prepend a summary-length token
3. Prepend a masked summary (UMR-S)

## Prepare training data from CNN/DM
Our best generation result is obtained with CNN/DM data. To train MargeSum on CNN/DM data, apart from the above-mentioned three customizations, we need an extra step: build a multi-document version of CNN/DM. 

This is mainly because the summaries in the original CNN/DM are fairly short, while testing on QFS requires 250 words as output. To fix this issue, we concatenate  summaries from a couple of relevant samples to get a long enough summary. Therefore, the input is now a cluster of the documents from these relevant samples. 

This involves in [Dr.QA](https://github.com/facebookresearch/DrQA) to index all summaries in CNN/DM. After indexing, you can use the following script to cluster samples via retrieving similar summaries:
```Python
python scripts/build_cnndm_clusters.py
```


## Inference and evaluation
### Setting up UniLM environment
To evaluate abstractive summarization, you need to setup an UniLM evironment following the instructions [here](https://github.com/microsoft/unilm/tree/master/s2s-ft). 

After setting up UnILM, in `src/frame/rr/main.py`, run:
```python
build_unilm_input(src='rank')
```
This turns ranked evidence from Marge into MargeSum input files. 

Now You can evaluate the trained UniLM model for developement and testing. Go to the UniLM project root, set the correct input directory, and deocode the summaries.

To evaluate the output, use the following function in `src/frame/rr/main.py`:
### 
```python
eval_unilm_out()
```

You can specifiy inference configs in `src/frame/rr/rr_config.py.`
