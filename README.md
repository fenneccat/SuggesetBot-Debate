# suggestbot-demo

## Requirements for DocumentRetriever

* `pip install -r requirements.txt`
* `python -m spacy download en_core_web_md`
* `python -m textblob.download_corpora`

## SentenceSelector

* Should be implemented inside `helper.sentence_selector`
* Input/Output: `(claim, [sent1, sent2, ...])` -> `[(relevant_sent_a, score), (relevant_sent_b, score), ...]`
  * Output should be sorted by the score in reverse order

## StanceClassifier

* Should be implemented inside `helper.stance_classifier`
* Input/Output: `(claim, [sent1, sent2, ...])` -> `[(sent1, 'SUPPROT'), (sent2, 'SUPPROT'), (sent3, 'REFUTE'), ...]`
  * Output should be sorted by the stance
