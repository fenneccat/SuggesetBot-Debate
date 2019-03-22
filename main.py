#!/usr/bin/env python
# coding: utf-8

from helper import DocumentRetriever

if __name__ == '__main__':
    doc_retriever = DocumentRetriever()
    with open('./data/queries_full.txt', encoding='utf-8-sig') as fin:
        texts = [line.strip() for line in fin]
        for text in texts:
            results = doc_retriever.search(text)




