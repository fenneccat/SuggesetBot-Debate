#!/usr/bin/env python
# coding: utf-8

from helper import SentenceRetriever
from helper import SentenceSelector ## Trained by Kialo dataset (recommended)
#from helper import SentenceSelectorPytorch as SentenceSelector   ## Trained by FEVER+IBM dataset sentence selector
from helper import StanceClassifier
from pprint import pprint


HOST = 'localhost'
PORT = 9200
INDEX = 'kixx' # Use hand-annotated documents (small - 46 docs)  ## Kix crowdsourcing document set
# INDEX = 'news-please' # Use crawled documents (big - 10,000,000)  ## large newspaper document set
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

    claim = "Taking in refugees costs a lot of money."
    print('Get evidence candidates for the claim: "{}"'.format(claim))
    candidates = get_candidates(claim, doc_k=5, sent_k=30)
    # candidates = [(sent_1, sent_1_orig, doc_id, doc_url), (sent_2, sent_2_orig, doc_id, doc_url), ...]
    pprint(candidates)

    # candidates_id_url_map -> {
    #   'sent_1': (doc_id, doc_url),
    #   'sent_2': (doc_id, doc_url),
    #   ...
    # }
    candidates_id_url_map = {c[0]: (c[2], c[3]) for c in candidates}
    candidates_text_only = list(candidates_id_url_map.keys())

    # 1. Select evidence sentences
    print('Evidences with relevancy_score')
    claim, ranked_evidences = sentence_selector.get_evidences(claim, candidates_text_only, k=10)
    pprint(ranked_evidences)

    # 2. Classify stances for the selected evidences
    print('Evidences with stance_score')
    ranked_evidences_text_only = [ev for ev, _ in ranked_evidences] # classify stances for evidences only
    claim, evidence_stances = stance_classifier.get_evidence_stance(claim, ranked_evidences_text_only)
    pprint(evidence_stances)
