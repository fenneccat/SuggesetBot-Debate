#!/usr/bin/env python
# coding: utf-8

from helper import DocumentRetriever
# from helper import SentenceSelector
from helper import SentenceSelectorPytorch as SentenceSelector
from helper import StanceClassifier
from pprint import pprint


HOST = 'localhost'
PORT = 9200
# INDEX = 'kixx' # 수동 태깅된 문서 사용 (소규모 - 46건)
INDEX = 'news-please' # 자동 크롤된 문서 사용 (대규모 - 100만건)
FIELDS = ['title', 'text']

# Initialize modules
document_retriever = DocumentRetriever(
    hosts=HOST, port=PORT, index=INDEX, fields=FIELDS)
sentence_selector = SentenceSelector()
stance_classifier = StanceClassifier()

def get_candidates(claim, k):
    docs = document_retriever.search(claim, k=k)['docs']
    candidates = []
    for d in docs:
        candidates += d['text']
    return candidates


if __name__ == '__main__':

    # sample claim
    claim = "Resettlement is needed now, because an organized, legal route to hope in the U.S., as well as in Canada and Australia, will disempower the smugglers who are currently charging 1,200 euros for desperate people to get the six kilometers from Turkey to Greece.".strip()
    print('Get evidence candidates for the claim: "{}"'.format(claim))
    candidates = get_candidates(claim, k=10)

    # Print evidences with stances
    print('Evidences with stances')
    # 1. Select evidence sentences
    claim, ranked_evidences = sentence_selector.get_evidences(claim, candidates, k=10)
    pprint(ranked_evidences)

    # 2. Classify stances for the selected evidences
    evidences = [ev for ev, _ in ranked_evidences] # classify stances for evidences only
    claim, evidence_stances = stance_classifier.get_evidence_stance(claim, evidences)
    pprint(evidence_stances)