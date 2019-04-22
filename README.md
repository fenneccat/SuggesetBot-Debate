# suggestbot-demo

## Before Cloning This Repo

* Please install **Git Large File Storage (LFS)** [here](https://git-lfs.github.com/) first before cloning this repository. It's required to download the models. If not, you will have only empty models, which will cause erros.

## Setup

### Install Dependencies

* `pip install -r requirements.txt`

### Install Search Engine (Elasticsearch) & Insert Evidence Documents

* Install Elasticsearch package at https://www.elastic.co/
  * Use default settings (`host=localhost, port=9200`)
* `python scripts/insert_kixx.py` and `python scripts/insert_news_please.py` to insert documents
  * Elasticsearch should be running

### Install Models

* `python -m spacy download en_core_web_md`
* `python -m textblob.download_corpora`

## Example (`main.py`)

```python
from helper import SentenceRetriever
from helper import SentenceSelectorPytorch as SentenceSelector
from helper import StanceClassifier
from pprint import pprint

HOST = 'localhost'
PORT = 9200
INDEX = 'kixx' # Use hand-annotated documents (small - 46 docs)
# INDEX = 'news-please' # Use crawled documents (big - 10,000,000)
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
```

## API

* Based on https://github.com/google-research/bert

### SentenceSelector

* How to use SentenceSelector
   * Generate SentenceSelector instance
      * `selector = SentenceSelector()`
   * call `get_evidences` method with claim, sentence candidates, and an optional values k which specifies desired number of evidences to retrieve (default is 5).
      * `claim, ranked_evidence = selector.get_evidence(claim, sentence_candidates)`
   * `ranked_evidence` is sorted based on confidence score in reverse order
* Input/Output: `(claim, [sent1, sent2, ...])` -> `(claim, [(relevant_sent_a, score), (relevant_sent_b, score), ...])`
  * Output should be sorted by the score in reverse order

### StanceClassifier

* How to use StanceClassifier
   * Generate StanceClassifier instance
      * `stance_classifier = StanceClassifier()`
   * call `get_evidence_stance` method with claim, sentence candidates.
      * `evidence_stance = stance_classifier.get_evidence_stance(claim, sentence_candidates)`
   * `evidence_stance` is sorted based on confidence score in reverse order
* Input/Output: `(claim, [sent1, sent2, ...])` -> `[(sent1, 'SUPPROTS', score), (sent2, 'SUPPROTS', score), (sent3, 'REFUTES', score), ...]`
  * Output is sorted by stance and confidence score of each stance (leftmost-strongest SUPPORTS, rightmost-strongest REFUTES)
