import os
import random
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pickle
from sql_metadata import Parser

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def create_directory(path):
  if not os.path.exists(path):
    os.makedirs(path)

def read_json(fn):
  with open(fn) as f:
    return json.load(f)

def write_json(obj, fn):
  with open(fn, 'w') as f:
    json.dump(obj, f, indent=2)

def read_pickle(fn):
  with open(fn, 'rb') as f:
    return pickle.load(f)

def write_pickle(r, fn):
  with open(fn, 'wb') as f:
    pickle.dump(r, f)

def mean_pooling(token_embeddings, mask):
  token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
  sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
  return sentence_embeddings

BATCH_SIZE = 200
def embed(texts, fn, hide_progress=False):
  if fn is not None and os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))
  
  tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
  model = AutoModel.from_pretrained('facebook/contriever-msmarco').cuda()

  embeds = []
  for i in tqdm(range((len(texts)//BATCH_SIZE) + 1), disable=hide_progress):
    _texts = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    if len(_texts) == 0:
      break
    assert len(_texts) >= 1
    
    inputs = tokenizer(_texts, padding=True, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
      vec = model(**inputs)
    vec = mean_pooling(vec[0], inputs['attention_mask'])

    embeds.append(vec.cpu().numpy())
  
  embeds = np.vstack(embeds)

  if fn is not None:
    np.save(fn, embeds)
  
  embeds = torch.from_numpy(embeds)

  return embeds

def get_corpus(dataset):
  tables = read_json(f'./data/{dataset}/dev_tables.json')
  return list(tables.keys())

def sql_to_tables(sql: str, db_id: str):
  gold_ts = Parser(sql).tables
  gold_ts = [f'{db_id}#sep#{gold_t}' for gold_t in gold_ts]
  return gold_ts

# format should be either json or npy
def merge(num_partitions: int, _fn: str, format: str):
  fn = f'{_fn}.{format}'

  results = []

  individuals = [read_json(f'{_fn}_{partition}.json') if format == 'json' else np.load(f'{_fn}_{partition}.npy') for partition in range(num_partitions)]

  if format == 'json':
    for result in individuals:
      results += result
    write_json(results, fn)
  elif format == 'npy':
    results = np.vstack(individuals)
    np.save(fn, results)
  
  print(len(results))

def get_skip_idxs(dataset: str):
  qs = read_json(f'./data/{dataset}/dev.json')
  skip_idxs = [i for i in range(len(qs)) if len(sql_to_tables(qs[i]['sql'], qs[i]['db_id'])) == 1]
  return skip_idxs