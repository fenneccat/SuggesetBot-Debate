from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import spacy
from elasticsearch import Elasticsearch
import json
from textblob import TextBlob
from pathlib import Path
from gensim.models import KeyedVectors
import logging
import re
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARN)

class SentenceRetriever:
    
    def __init__(self, hosts='localhost', port=9200, index='kixx', fields=['title', 'text']):
        # init logger
        self.logger = logging.getLogger(__name__)
        # load model
        model_name='tfidf_2019-03-18T15-53-19'
        self._load_model(model_name)
        # init search interface
        self._init_search(hosts, port, index, fields)
        
    def search(self, text, doc_k, sent_k):
        # build query
        # root_topic_tokens = ['Syria', 'refugee']
        # q = self._extract_important_words(text) + root_topic_tokens
        # q = ' '.join(q)
        q = text + ' ' + 'syria refugee'
        Q = self._build_query(q, fields=self.fields, limit=doc_k)
        res = self.es.search(index=self.index, body=Q)
        docs = [hit["_source"] for hit in res['hits']['hits']]

        # split document into sentences
        sentences_with_score = []
        for d in docs:
            doc_id, url = d['doc_id'], d['doc_url']
            sentences = self._split_sentences(d['text'])
            for sent in sentences:
                score = self._get_wmd_score(q, sent)
                sentences_with_score.append((sent, score, doc_id, url))
        sentences_with_score = sorted(sentences_with_score, key=lambda item: item[1], reverse=True)
        result = self._prepare_result(sentences_with_score, sent_k)
        return result

    def _prepare_result(self, sentences_with_score, sent_k):
        # sentences = [sent for sent, _ in sentences_with_score]
        # 1. remove duplicates
        # 2. length > 5 words
        # 3. return only sent_k
        MIN_LEN = 5
        result = []
        for sent, score, doc_id, url in sentences_with_score:
            # clean sentence
            sent_cleaned = self._clean_sentence(sent)
            if (sent not in result) and (len(sent.split()) > MIN_LEN):
                result.append((sent_cleaned, sent, doc_id, url))
                if len(result) >= sent_k:
                    break
        return result

    def _clean_sentence(self, sentence):
        # remove citation & newline
        sentence = re.sub('(\[\d+\]|\n)', ' ', sentence)
        return sentence

    def _get_wmd_score(self, query, sentence):
        query = [w.lower() for w in TextBlob(query).words]
        sent = [w.lower() for w in TextBlob(sentence).words]
        return (sent, self.embeddings.wmdistance(query, sent))
           
    def _init_search(self, hosts, port, index, fields):
        self.es = Elasticsearch(hosts=hosts, port=port)
        self.index = index
        self.fields = fields

    def _build_query(self, query, fields, limit):
        return {
            "query": {
                "multi_match": {
                    "query": query,
                    "type": "most_fields",
                    "fields": fields,
                    "tie_breaker": 0.3
                }
            },
            "from" : 0, "size" : limit,
        }

    def _split_sentences(self, text):
        return [str(sent) for sent in TextBlob(text).sentences]
        
    def _load_model(self, model_name):
        self.logger.warn('Loading DocumentRetriever models...')
        model_dir = Path('./model')
        self.dct = Dictionary.load(str(model_dir / f'{model_name}.dict'))
        self.tfidf = TfidfModel.load(str(model_dir / f'{model_name}.tfidf'))
        self.nlp = spacy.load('en_core_web_md')
        self.embeddings = KeyedVectors.load(str(model_dir / 'wiki-news-300d-1M-subword'))
        
    def _get_tags(self, spacy_doc):
        return [(tok, tok.pos_) for tok in spacy_doc]
    
    def _extract_important_words(self, text, k=10, tfidf_thresh=0.2):
        tokens = []
        dct, tfidf = self.dct, self.tfidf
        nlp = self.nlp

        # filter by tag
        doc = nlp(text)
        token_tags = self._get_tags(doc)
        target_tags = ('NOUN', 'VERB', 'PROPN')
        custom_stops = ("'re", "'s", "’s", "'re")
        tokens += [token.text for (token, tag) in token_tags 
                  if tag in target_tags and not token.is_stop and token.text not in custom_stops]

        # filter by entity
        entities = [ent.text for ent in doc.ents if ent.label_ not in ('QUANTITY', 'ORDINAL', 'CARDINAL')]
        tokens += entities

        # calculate tf-idf
        bow = dct.doc2bow(tokens)
        word_scores = tfidf[bow]
        top_k = sorted(word_scores, key=lambda pair: pair[1], reverse=True)
        return [dct[idx] for idx, score in top_k if score > tfidf_thresh]
