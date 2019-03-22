from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np
import spacy
from elasticsearch import Elasticsearch
import json
from textblob import TextBlob
from pathlib import Path
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARN)

class DocumentRetriever:
    
    def __init__(self, model_name='tfidf_2019-03-18T15-53-19'):
        # init logger
        self.logger = logging.getLogger(__name__)
        # load model
        self._load_model(model_name)
        # init search interface
        self._init_search()
        
    def search(self, text):
        INDEX = 'news-please'
        root_topic_tokens = ['Syria', 'refugee']
        dct, tfidf = self.dct, self.tfidf
        
        # build query
        q = self._extract_important_words(text) + root_topic_tokens
        Q = self._build_query(' '.join(q))
        res = self.es.search(index=INDEX, body=Q)
        docs = [hit["_source"] for hit in res['hits']['hits']]

        # split document into sentences
        docs_splitted = []
        for d in docs:
            d = d.copy()
            d['text'] = self._get_sentences(d['text'])
            docs_splitted.append(d)

        # show
        # print("Text:", text)
        # print('Query:', q)
        # print([d['title'] for d in docs])
        # print('')

        # save results
        results = []
        results.append({
            'text': text,
            'query': ' '.join(q),
            'docs': docs_splitted
        })

        return results
           
    def _init_search(self, hosts=['server.kyoungrok.kr'], port=19200):
        self.es = Elasticsearch(hosts=hosts, port=port)

    def _build_query(self, query, fields=['title', 'text', 'description'], limit=10):
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

    def _get_sentences(self, text):
        return [str(sent) for sent in TextBlob(text).sentences]
        
    def _load_model(self, model_name):
        self.logger.warn('Loading DocumentRetriever models...')
        model_dir = Path('./model')
        self.dct = Dictionary.load(str(model_dir / f'{model_name}.dict'))
        self.tfidf = TfidfModel.load(str(model_dir / f'{model_name}.tfidf'))
        self.nlp = spacy.load('en_core_web_md')
        
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