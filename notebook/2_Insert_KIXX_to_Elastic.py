#!/usr/bin/env python
# coding: utf-8

# In[11]:


from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pathlib import Path


# In[32]:


INDEX = 'kixx'
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


def get_action(title, text, index):
    return {
        '_index': index,
        '_type': 'evidence',
        '_source': {
            'title': title,
            'text': text
        }
    }


# In[35]:


print('Inserting evidence documents...')
files = Path('../data/kixx').glob('*_sp.txt')
actions = []
for f in files:
    with f.open(encoding='utf-8-sig') as fin:
        title, text = fin.name.split('\\')[-1], fin.read()
        action = get_action(title, text, INDEX)
        actions.append(action)
        
res = bulk(es, actions)
print(f'Inserted {res[0]} documents.')


# In[ ]:




