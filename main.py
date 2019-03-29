#!/usr/bin/env python
# coding: utf-8

from helper import DocumentRetriever
from helper import SentenceSelector
from pprint import pprint


HOST = 'localhost'
PORT = 9200
INDEX = 'kixx'
FIELDS = ['title', 'text']
document_retriever = DocumentRetriever(
    hosts=HOST, port=PORT, index=INDEX, fields=FIELDS)
selector = SentenceSelector()


def get_candidates(claim, k):
    docs = document_retriever.search(claim, k=k)['docs']
    candidates = []
    for d in docs:
        candidates += d['text']
    return candidates


if __name__ == '__main__':

    # test
    claim = "Resettlement is needed now, because an organized, legal route to hope in the U.S., as well as in Canada and Australia, will disempower the smugglers who are currently charging 1,200 euros for desperate people to get the six kilometers from Turkey to Greece.".strip()
    print('Get evidences for the claim: "{}"'.format(claim))
    candidates = get_candidates(claim, k=5)
    claim, ranked_evidences = selector.get_evidences(claim, candidates, k=5)
    pprint(ranked_evidences)
