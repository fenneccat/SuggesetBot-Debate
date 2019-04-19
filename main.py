#!/usr/bin/env python
# coding: utf-8

from helper import SentenceRetriever
# from helper import SentenceSelector
from helper import SentenceSelectorPytorch as SentenceSelector
from helper import StanceClassifier
from pprint import pprint


HOST = 'localhost'
PORT = 9200
# INDEX = 'kixx' # Use hand-annotated documents (small - 46 docs)
INDEX = 'news-please' # Use crawled documents (big - 10,000,000)
FIELDS = ['title', 'text']

# Initialize modules
sentence_retriever = SentenceRetriever(
    hosts=HOST, port=PORT, index=INDEX, fields=FIELDS)
sentence_selector = SentenceSelector()
stance_classifier = StanceClassifier()

def get_candidates(claim, doc_k, sent_k):
    candidates = sentence_retriever.search(claim, doc_k=doc_k, sent_k=sent_k)
    return candidates


if __name__ == '__main__':

    # sample claimS
    # 1. If Australia can take 18,000 refugees, if Canada can take 25,000, if France can take 35,000, then the United States, 10 times the size of Canada, can take 100,000 refugees.
    # 2. It takes 18 to 24 months for the average Syrian refugee to get through the security screening process.
    # 3. Most the refugees are in the Middle East. And that's where most of the help is going.
    # 4. Almost two-thirds of Germans say their country has accepted too many refugees.
    # 5. Accepting refugees benefit the citizens of high-income countries.
    # 6. Taking in refugees costs a lot of money.
    # 7. All high-income countries are signatories to the 1951 Refugee Convention.
    # 8. Taking in refugees will increase criminality within host countries.
    # 9. Most refugees arriving in high-income countries are Muslims; their cultural and religious backgrounds have led to many conflicts in the past.
    # 10. Accepting refugees can be a solution to the problem of aging populations.

    claim = "Taking in refugees will increase criminality within host countries."
    print('Get evidence candidates for the claim: "{}"'.format(claim))
    candidates = get_candidates(claim, doc_k=20, sent_k=20)
    pprint(candidates)

    # Print evidences with stances
    # print('Evidences with stances')
    # # 1. Select evidence sentences
    # claim, ranked_evidences = sentence_selector.get_evidences(claim, candidates, k=10)
    # pprint(ranked_evidences)

    # # 2. Classify stances for the selected evidences
    # evidences = [ev for ev, _ in ranked_evidences] # classify stances for evidences only
    # claim, evidence_stances = stance_classifier.get_evidence_stance(claim, evidences)
    # pprint(evidence_stances)