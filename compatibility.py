import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sqlite3
import os

from utils import read_json, write_json, embed

def compute_jaccard(dataset: str):
  tables = read_json(f'./data/{dataset}/dev_tables.json')

  jaccard = {}

  fn = f'./data/{dataset}/dev_jaccard.json'
  if os.path.isfile(fn):
    jaccard = read_json(fn)

  for t1 in tqdm(tables):
    for t2 in tqdm(tables, disable=dataset == 'spider'):
      if t1 == t2:
        continue

      if f'{t1}-{t2}' in jaccard or f'{t2}-{t1}' in jaccard:
        continue
      
      table_pair_key = f'{t1}-{t2}'
      jaccard[table_pair_key] = {}

      db1, db2 = tables[t1]['db_id'], tables[t2]['db_id']
      t_name1, t_name2 = tables[t1]['table_name_original'], tables[t2]['table_name_original']

      conn1 = sqlite3.connect(f'./data/{dataset}/dev_database/{db1}/{db1}.sqlite')
      conn2 = sqlite3.connect(f'./data/{dataset}/dev_database/{db2}/{db2}.sqlite')
      cur1, cur2 = conn1.cursor(), conn2.cursor()
    
      for c1 in tables[t1]['column_names_original']:
        cur1.execute(f'select `{c1}` from `{t_name1}`')
        r1 = cur1.fetchall()
        r1 = set(x[0] for x in r1)

        for c2 in tables[t2]['column_names_original']:
          cur2.execute(f'select `{c2}` from `{t_name2}`')
          r2 = cur2.fetchall()
          r2 = set(x[0] for x in r2)
          
          sim_score = len(r1 & r2) / len(r2 | r2)

          col_pair_key = f'{t1}#sep#{c1}-{t2}#sep#{c2}'
          jaccard[table_pair_key][col_pair_key] = sim_score
      
      conn1.close()
      conn2.close()
  
    write_json(jaccard, fn)

def get_uniqueness(dataset: str):
  tables = read_json(f'./data/{dataset}/dev_tables.json')
  
  u_scores = {}

  for t in tqdm(tables):
    db, t_name = tables[t]['db_id'], tables[t]['table_name_original']

    conn = sqlite3.connect(f'./data/{dataset}/dev_database/{db}/{db}.sqlite')
    cur = conn.cursor()

    for c in tables[t]['column_names_original']:
      cur.execute(f'select `{c}` from `{t_name}`')
      r_orig = cur.fetchall()
      r = set(x[0] for x in r_orig)
    
      u_scores[f'{t}#sep#{c}'] = len(r) / len(r_orig)
  
  write_json(u_scores, f'./data/{dataset}/dev_uniqueness.json')

def process_word_fast(word: str):
  return word.replace('_', ' ').replace('.', '').lower().strip()  

def overlap_coefficient(s1: str, s2: str):
  s1, s2 = process_word_fast(s1).split(' '), process_word_fast(s2).split(' ')

  s1, s2 = set(s1), set(s2)
  return len(s1 & s2) / min(len(s1), len(s2))

# the serialized format is 'db_id table_name column_name'
def get_cols_embeds(dataset):
  tables = read_json(f'./data/{dataset}/dev_tables.json')

  cols, cols_idxs = [], [0]

  for t in tables:
    db, t_name = tables[t]['db_id'], tables[t]['table_name_original']
    t_cols = [process_word_fast(f'{db} {t_name} {c}') for c in tables[t]['column_names_original']]
    cols += t_cols
    cols_idxs.append(cols_idxs[-1] + len(t_cols))

  cols_embeds = embed(cols, None, hide_progress=True)
  cols_embeds_dict = {}

  for t_idx, t in enumerate(tables):
    cols_embeds_dict[t] = cols_embeds[cols_idxs[t_idx]:cols_idxs[t_idx + 1]]

  return cols_embeds_dict

def get_col_sim(dataset):
  tables = read_json(f'./data/{dataset}/dev_tables.json')

  exact_sim, semantic_sim = {}, {}
  exact_sim_fn = f'./data/{dataset}/exact_col_sim.json'
  semantic_sim_fn = f'./data/{dataset}/semantic_col_sim.json'

  cols_embeds = get_cols_embeds(dataset)

  for t1 in tqdm(tables):
    for t2 in tables:
      if t1 == t2:
        continue

      if f'{t1}-{t2}' in exact_sim or f'{t2}-{t1}' in exact_sim:
        continue
            
      table_pair_key = f'{t1}-{t2}'
      exact_sim[table_pair_key], semantic_sim[table_pair_key] = {}, {}

      db1, db2 = tables[t1]['db_id'], tables[t2]['db_id']
      t_name1, t_name2 = tables[t1]['table_name_original'], tables[t2]['table_name_original']
      cols1_embeds, cols2_embeds = cols_embeds[t1], cols_embeds[t2]

      semantic_sim_matrix = []
      for x in cols1_embeds:
        semantic_sim_matrix.append(F.cosine_similarity(x.unsqueeze(0), cols2_embeds, dim=1).unsqueeze(0))
      semantic_sim_matrix = torch.vstack(semantic_sim_matrix).tolist()

      for i1, c1 in enumerate(tables[t1]['column_names_original']):
        for i2, c2 in enumerate(tables[t2]['column_names_original']):
          col_pair_key = f'{t1}#sep#{c1}-{t2}#sep#{c2}'

          exact_score = overlap_coefficient(f'{db1} {t_name1} {c1}', f'{db2} {t_name2} {c2}')
          exact_sim[table_pair_key][col_pair_key] = exact_score
          semantic_sim[table_pair_key][col_pair_key] = semantic_sim_matrix[i1][i2]
  
    write_json(exact_sim, exact_sim_fn)
    write_json(semantic_sim, semantic_sim_fn)

def get_score(t1, col1, t2, col2, score_dict):
  if f'{t1}-{t2}' in score_dict:
    score = score_dict[f'{t1}-{t2}']
  else:
    score = score_dict[f'{t2}-{t1}']

  if f'{t1}#sep#{col1}-{t2}#sep#{col2}' in score:
    score = score[f'{t1}#sep#{col1}-{t2}#sep#{col2}']
  else:
    score = score[f'{t2}#sep#{col2}-{t1}#sep#{col1}']
  
  return score

# ts should be a list of table names
def get_cr(dataset, ts):
  d = read_json(f'./data/{dataset}/dev_uniqueness.json')
  semantic_col_sim = read_json(f'./data/{dataset}/semantic_col_sim.json')
  exact_col_sim = read_json(f'./data/{dataset}/exact_col_sim.json')
  jaccard = read_json(f'./data/{dataset}/dev_jaccard.json')

  cr = {}

  tables = read_json(f'./data/{dataset}/dev_tables.json')

  for i, t1 in enumerate(ts):
    cols1, cols1_type = tables[t1]['column_names_original'], tables[t1]['column_types']

    for j, t2 in enumerate(ts):
      cols2, cols2_type = tables[t2]['column_names_original'], tables[t2]['column_types']
      
      cr[(i, j)] = np.zeros((len(cols1), len(cols2)))
      
      if i == j:
        continue
    
      for k, col1 in enumerate(cols1):
        for l, col2 in enumerate(cols2):
          u_score = max(d[f'{t1}#sep#{col1}'], d[f'{t2}#sep#{col2}'])
          # two columns can only join if they are of the same type and if at least one column is unique
          if cols1_type[k] == cols2_type[l] and u_score != 0:
            cr[(i, j)][k][l] += 0.5 * get_score(t1, col1, t2, col2, jaccard)
            cr[(i, j)][k][l] += 0.5 * (0.5 * get_score(t1, col1, t2, col2, semantic_col_sim) + 0.5 * get_score(t1, col1, t2, col2, exact_col_sim))
            cr[(i, j)][k][l] *= u_score

  return cr