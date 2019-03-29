# suggestbot-demo

## Before Cloning This Repo

* Please install Git Large File Storage (LFS) [here](https://git-lfs.github.com/) first before cloning this repository. It's required to download the models. If not, you will have only empty models, which will cause erros.

## Setup

### Install Dependencies

* `pip install -r requirements.txt`

### Install Search Engine (Elasticsearch) & Insert Evidence Documents

* Install Elasticsearch package at https://www.elastic.co/
  * Use default settings (`host=localhost, port=9200`)
* `python scripts/2_Insert_KIXX_to_Elastic.py` to insert documents
  * Elasticsearch should be running

### Install Models

* `python -m spacy download en_core_web_md`
* `python -m textblob.download_corpora`

## Example

```python
from helper import DocumentRetriever
from helper import SentenceSelector
from pprint import pprint


HOST = 'localhost'
PORT = 9200
INDEX = 'kixx'
FIELDS = ['title', 'text']
document_retriever = DocumentRetriever(hosts=HOST, port=PORT, index=INDEX, fields=FIELDS)
selector = SentenceSelector()

def get_candidates(claim):
    docs = document_retriever.search(claim)['docs']
    candidates = []
    for d in docs:
        candidates += d['text']
    return candidates

if __name__ == '__main__':
    
    # test
    claim = """
        Resettlement is needed now, because an organized, legal route to hope in the U.S., as well as in Canada and Australia, will disempower the smugglers who are currently charging 1,200 euros for desperate people to get the six kilometers from Turkey to Greece.
    """
    candidates = get_candidates(claim)
    claim, ranked_evidences = selector.get_evidences(claim, candidates)
    pprint(ranked_evidences)

```

## API

### SentenceSelector

* Depends on https://github.com/huggingface/pytorch-pretrained-BERT
* How to use SentenceSelector
   * Generate SentenceSelector instance
      * selector = SentenceSelector()
   * call `get_evidences` method with claim and sentence candidates
      * claim, ranked_evidnce = selector.get_evidence(claim, sentence_candidates)
   * `ranked_evidence` is sorted based on confidence score in reverse order
* Input/Output: `(claim, [sent1, sent2, ...])` -> `(claim, [(relevant_sent_a, score), (relevant_sent_b, score), ...])`
  * Output should be sorted by the score in reverse order

### StanceClassifier (TODO)

* Should be implemented inside `helper.stance_classifier`
* Input/Output: `(claim, [sent1, sent2, ...])` -> `[(sent1, 'SUPPROT'), (sent2, 'SUPPROT'), (sent3, 'REFUTE'), ...]`
  * Output should be sorted by the stance
