#!/usr/bin/env python
# coding: utf-8

# In[11]:


from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pathlib import Path
import pandas as pd
from tqdm import tqdm


# In[32]:


INDEX = 'news-please'
es = Elasticsearch()


# In[33]:


def reset_index(index):
    if es.indices.exists(index):
        es.indices.delete(index)
    es.indices.create(index)
    # set English analyzer
    settings = {
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "english"
                }
            }
        }
    }
    es.indices.close(index)
    es.indices.put_settings(body=settings, index=index)
    es.indices.open(index)
    
reset_index(INDEX)


# In[34]:


def get_action(title, text, url, index):
    return {
        '_index': index,
        '_type': 'evidence',
        '_source': {
            'title': title,
            'text': text,
            'doc_id': url,
            'doc_url': url
        }
    }


# In[35]:


print('Inserting evidence documents...')
fpath = Path('./data/news-please/news-please.parquet')
df = pd.read_parquet(fpath)
actions = []
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    title, text = row['title'] + (row['description'] or ''), row['text']
    url = row['url']
    action = get_action(title, text, url, INDEX)
    actions.append(action)
res = bulk(es, actions)
print(f'Inserted {res[0]} documents.')


# In[ ]:




