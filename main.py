#!/usr/bin/env python
# coding: utf-8

from helper import DocumentRetriever
from pprint import pprint

if __name__ == '__main__':
    doc_retriever = DocumentRetriever(hosts=['localhost'], port=9200, index='kixx', fields=['title', 'text'])
    with open('./data/queries_full.txt', encoding='utf-8-sig') as fin:
        texts = [line.strip() for line in fin]
        for text in texts:
            results = doc_retriever.search(text)
            pprint(results)
