# suggestbot-demo

## Before Cloning This Repo

* Please install **Git Large File Storage (LFS)** [here](https://git-lfs.github.com/) first before cloning this repository. It's required to download the models. If not, you will have only empty models, which will cause erros.

## Setup

### Install Dependencies

* `pip install -r requirements.txt`

### Install Search Engine (Elasticsearch) & Insert Evidence Documents

* Install Elasticsearch package at https://www.elastic.co/
  * Use default settings (`host=localhost, port=9200`)
* `python scripts/insert_kixx.py` to insert documents
  * Elasticsearch should be running

### Install Models

* `python -m spacy download en_core_web_md`
* `python -m textblob.download_corpora`

## Example (`main.py`)

```python
from helper import DocumentRetriever
# from helper import SentenceSelector
from helper import SentenceSelectorPytorch as SentenceSelector
from helper import StanceClassifier
from pprint import pprint


HOST = 'localhost'
PORT = 9200
INDEX = 'kixx'
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
    # pprint(ranked_evidences)

    # 2. Classify stances for the selected evidences
    evidences = [ev for ev, _ in ranked_evidences] # classify stances for evidences only
    claim, evidence_stances = stance_classifier.get_evidence_stance(claim, evidences)
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
