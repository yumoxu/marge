# -*- coding: utf-8 -*-
from os import listdir
import numpy as np
from os.path import dirname, abspath
import sys

from utils.config_loader import config_model
from data.clip_and_mask import mask_para
import utils.tools as tools

sys.path.insert(0, dirname(dirname(abspath(__file__))))


def build_and_mask_doc_embeds(embed_doc_fp):
    # build para x
    d_embed_univ = config_model['d_embed_univ']
    max_ns_doc = config_model['max_ns_doc']

    doc_sent_embeds = tools.load_obj(embed_doc_fp)

    doc_sent_embeds = doc_sent_embeds[:max_ns_doc]
    doc_masks = mask_para(len(doc_sent_embeds), max_n_sents=max_ns_doc)

    if len(doc_sent_embeds) < max_ns_doc:
        pad = np.zeros([max_ns_doc-len(doc_sent_embeds), d_embed_univ], dtype=np.float32)
        doc_sent_embeds = np.concatenate([doc_sent_embeds, pad])

    xx_doc = {
        'sent_embeds': doc_sent_embeds,
        'doc_masks': doc_masks,  # max_ns_doc,
    }

    return xx_doc


def build_and_mask_trigger_embeds(embed_trigger_fp):
    # build para x
    d_embed_univ = config_model['d_embed_univ']
    max_ns_trigger = config_model['max_ns_trigger']

    trigger_sent_embeds = tools.load_obj(embed_trigger_fp)

    trigger_sent_embeds = trigger_sent_embeds[:max_ns_trigger]
    trigger_mask = mask_para(len(trigger_sent_embeds), max_n_sents=max_ns_trigger)

    if len(trigger_sent_embeds) < max_ns_trigger:
        pad = np.zeros([max_ns_trigger-len(trigger_sent_embeds), d_embed_univ], dtype=np.float32)
        trigger_sent_embeds = np.concatenate([trigger_sent_embeds, pad])

    xx_trigger = {
        'trigger_embeds': trigger_sent_embeds,
        'trigger_masks': trigger_mask,  # max_ns_doc,
    }

    return xx_trigger


def build_and_mask_multi_trigger_embeds(embed_trigger_fps):
    xx_triggers = []
    for embed_trigger_fp in embed_trigger_fps:
        xx_trigger = build_and_mask_trigger_embeds(embed_trigger_fp)
        xx_triggers.append(xx_trigger)

    trigger_embeds = np.stack([xx_trigger['trigger_embeds'] for xx_trigger in xx_triggers])
    trigger_masks = np.stack([xx_trigger['trigger_masks'] for xx_trigger in xx_triggers])

    # print('trigger_embeds shape: {}'.format(trigger_embeds.shape))
    # print('trigger_masks shape: {}'.format(trigger_masks.shape))

    return {
        'trigger_embeds': trigger_embeds,
        'trigger_masks': trigger_masks,  # max_ns_doc,
    }
