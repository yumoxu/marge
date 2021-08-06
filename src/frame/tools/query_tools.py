# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import utils.config_loader as config
from utils.config_loader import path_parser, years
import json


def get_annual_masked_query(year, query_type, with_sub, concat_subq):
    """
        with_sub: use the prepared query that has token [SUBQUERY] in-between sentences.
        concat_subq: concat the subquery list and return a string
    """
    fp = path_parser.masked_query / f'{year}.json'
    annual_dict = dict()
    with open(fp) as f:
        for line in f:
            json_obj = json.loads(line.rstrip('\n'))
            cid = json_obj['cid']

            if with_sub:
                masked_query = json_obj['masked_query_with_sub']
            else:

                masked_query = json_obj['masked_query']
                # title = masked_query[0].strip() + '.'
                # narr = ' '.join(masked_query[1:])
                if query_type == config.TITLE:
                    masked_query = masked_query[0]
                elif query_type == config.NARR:
                    masked_query = masked_query[1:]
                    if concat_subq:
                        masked_query = ' '.join(masked_query[1:])
                elif query_type == config.QUERY:
                    if concat_subq:
                        masked_query = ' '.join(masked_query)
                else:
                    raise ValueError(f'Invalid query_type: {query_type}')

            print(f'masked_query: {masked_query}')
            annual_dict[cid] = masked_query
    return annual_dict


def get_annual_raw_query(year, query_type):
    """
        with_sub: use the prepared query that has token [SUBQUERY] in-between sentences.
        concat_subq: concat the subquery list and return a string
    """
    fp = path_parser.masked_query / f'{year}.json'
    annual_dict = dict()
    with open(fp) as f:
        for line in f:
            json_obj = json.loads(line.rstrip('\n'))
            cid = json_obj['cid']

            raw_query = json_obj['raw_query']
            title = raw_query[0].strip() + '.'
            narr = ' '.join(raw_query[1:])
            
            if query_type == config.TITLE:
                query = title
            elif query_type == config.NARR:
                query = narr
            elif query_type == config.QUERY:
                query = title + ' ' + narr
            else:
                raise ValueError(f'Invalid query_type: {query_type}')

            print(f'query: {query}')
            annual_dict[cid] = query
    return annual_dict


def get_cid2masked_query(query_type, with_sub, concat_subq=True):
    query_dict = dict()
    for year in years:
        annual_dict = get_annual_masked_query(year, 
            query_type=query_type, 
            with_sub=with_sub, 
            concat_subq=concat_subq)
        
        query_dict = {
            **annual_dict,
            **query_dict,
        }
    return query_dict


def get_cid2raw_query(query_type):
    query_dict = dict()
    for year in years:
        annual_dict = get_annual_raw_query(year, 
            query_type=query_type)
        
        query_dict = {
            **annual_dict,
            **query_dict,
        }
    return query_dict