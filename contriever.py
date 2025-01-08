import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from itertools import chain
import os

from utils import set_seed, read_json, embed, write_pickle, get_corpus, write_json
from metrics import eval_preds

def decompose_schema(tables):
  r, col_nums = [], []
  for t in tables:
    t_name, t_cols = tables[t]['table_name_original'], tables[t]['column_names_original']
    for t_col in t_cols:
      r.append(f'{t_name}:{t_col}')
    col_nums.append(len(t_cols))
  return r, col_nums

def ravel_t_score(score, col_nums):
  r = []
  idx = 0
  for col_num in col_nums:
    r.append(score[idx:idx+col_num])
    idx += col_num
  assert(idx == len(score))
  return r

def get_sim_scores(dataset, dq: bool, cols_nums=None):
  if dq:
    save_fn = f'./data/{dataset}/contriever/score_decomp.pkl'
    q_embeds_fn = f'./data/{dataset}/contriever/q_decomp.npy'
    t_embeds_fn = f'./data/{dataset}/contriever/t_decomp.npy'
  else:
    save_fn = f'./data/{dataset}/contriever/score.npy'
    q_embeds_fn = f'./data/{dataset}/contriever/q.npy'
    t_embeds_fn = f'./data/{dataset}/contriever/t.npy'

  q_embeds, t_embeds = torch.from_numpy(np.load(q_embeds_fn)), torch.from_numpy(np.load(t_embeds_fn))

  print(f'#q, #t: {q_embeds.shape[0]}, {t_embeds.shape[0]}')

  if not os.path.isfile(save_fn):
    sim_scores = []
    for q_embed in tqdm(q_embeds):
      sim_scores.append(F.cosine_similarity(q_embed.unsqueeze(0), t_embeds, dim=1).unsqueeze(0))
    sim_scores = torch.vstack(sim_scores).numpy()
    print(sim_scores.shape)
    
    if not dq:
      np.save(save_fn, sim_scores)
  else:
    sim_scores = np.load(save_fn)

  if dq:
    # all subqueries are flattened --> for each subquery, compute similarity to all columns in all tables
    sim_scores = [ravel_t_score(score, cols_nums) for score in tqdm(sim_scores)]
    write_pickle(sim_scores, save_fn)
  
  return sim_scores

def serialize_table(table):
  db_id, table_name, cols = table['db_id'], table['table_name_original'], table['column_names_original']
  return ' '.join([db_id, table_name] + cols)

def evaluate(dataset, model, k):
  corpus_tables = get_corpus(dataset)
  scores = torch.from_numpy(np.load(f'./data/{dataset}/{model}/score.npy'))
  top_idxs = scores.topk(k=k, dim=-1)[1]

  preds = []

  for top_idx in top_idxs:
    preds.append([corpus_tables[i] for i in top_idx])
  
  eval_preds(dataset, preds)

  write_json(preds, f'./data/{dataset}/{model}/preds_{k}.json')

if __name__ == '__main__':
  set_seed(1234)

  model = ['tapas', 'contriever'][1]
  dataset = ['bird', 'spider'][0]
  dq = True

  if dq:
    q_fn, t_fn = f'./data/{dataset}/contriever/q_decomp.npy', f'./data/{dataset}/contriever/t_decomp.npy'
  else:
    q_fn, t_fn = f'./data/{dataset}/contriever/q.npy', f'./data/{dataset}/contriever/t.npy'
  
  qs = read_json(f'./data/{dataset}/dev.json')
  qs = [q['question'] for q in qs]

  if dq:
    qs = read_json(f'./data/{dataset}/decomp.json')
    qs = list(chain.from_iterable(qs))

  tables = read_json(f'./data/{dataset}/dev_tables.json')
  
  if dq:
    ts, col_nums = decompose_schema(tables)
  else:
    ts, col_nums = [serialize_table(tables[t]) for t in tables], None
  
  # embed(qs, q_fn)
  # embed(ts, t_fn)
  # get_sim_scores(dataset, dq, cols_nums=col_nums)
  # evaluate(dataset, model, k=20)

  
