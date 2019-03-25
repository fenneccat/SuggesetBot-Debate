# suggestbot-demo

## Setup

### Install Search Engine (Elasticsearch) & Insert Evidence Documents

* Install Elasticsearch package at https://www.elastic.co/
  * Use default settings (`host=localhost, port=9200`)
* `python notebook/2_Insert_KIXX_to_Elastic.py` to insert documents

### Install Models

* `pip install -r requirements.txt`
* `python -m spacy download en_core_web_md`
* `python -m textblob.download_corpora`

## SentenceSelector

* Requirement for BERT is needed: https://github.com/huggingface/pytorch-pretrained-BERT
* How to use SentenceSelector
   * Generate SentenceSelector instance
      * selector = SentenceSelector()
   * call `get_evidence` method with claim and sentence candidates
      * claim, ranked_evidnce = selector.get_evidence(claim, sentence_candidates)
   * `ranked_evidence` is sorted based on confidence score in reverse order
* Input/Output: `(claim, [sent1, sent2, ...])` -> `(claim, [(relevant_sent_a, score), (relevant_sent_b, score), ...])`
  * Output should be sorted by the score in reverse order

## StanceClassifier

* Should be implemented inside `helper.stance_classifier`
* Input/Output: `(claim, [sent1, sent2, ...])` -> `[(sent1, 'SUPPROT'), (sent2, 'SUPPROT'), (sent3, 'REFUTE'), ...]`
  * Output should be sorted by the stance
