#!/usr/bin/env python
# coding: utf-8

# In[11]:


from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pathlib import Path
import pandas as pd


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


def get_action(title, text, ev_id, ev_url, index):
    return {
        '_index': index,
        '_type': 'evidence',
        '_source': {
            'title': title,
            'text': text,
            'doc_id': ev_id,
            'doc_url': ev_url
        }
    }


# In[35]:
df_url = pd.read_csv('./data/kixx-url.tsv', sep='\t').set_index('evidence_id')
def get_url(ev_id):
    try:
        return df_url.loc[ev_id].evidence_url
    except:
        return ''

print('Inserting evidence documents...')
files = Path('./data/kixx').glob('*_sp.txt')
print(len(list(files)))
actions = []
for f in files:
    with f.open(encoding='utf-8-sig') as fin:
        title, text = fin.name.split('\\')[-1], fin.read()
        ev_id = int(f.name.replace('_sp.txt', ''))
        ev_url = get_url(ev_id)
        action = get_action(title, text, ev_id, ev_url, INDEX)
        actions.append(action)

# print(actions[0])

res = bulk(es, actions)
print(f'Inserted {res[0]} documents.')


# In[ ]:




