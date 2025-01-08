from mip import *
import numpy as np
from tqdm import tqdm
import argparse

from utils import read_json, read_pickle, write_json, get_corpus, merge, get_skip_idxs
from compatibility import get_cr
from metrics import eval_preds

def assign_columns(lm, q_scores, dq_scores, dataset, k_num, join_num, t_num, w, num_partitions, partition, fn):
  w1, w2, w3, w4, w5 = w

  corpus_tables = get_corpus(dataset)
  qs = read_json(f'./data/{dataset}/dev.json')
  tables = read_json(f'./data/{dataset}/dev_tables.json')

  interval = len(qs) // num_partitions + 1
  start_idx, end_idx = partition * interval, (partition + 1) * interval

  fn = f'{fn}_{partition}.json'

  skip_idxs = get_skip_idxs(dataset)

  preds = []

  for q_idx, q in enumerate(tqdm(qs)):
    if not (start_idx <= q_idx < end_idx):
      continue

    if q_idx in skip_idxs:
      preds.append([])
      continue

    m = Model(sense=MAXIMIZE)

    score = q_scores[q_idx]

    top_idxs = np.argsort(-score)[:t_num]
    top_tables = [corpus_tables[top_idx] for top_idx in top_idxs]
    
    num_q = len(dq_scores[q_idx])

    # dq_scores
    # all subqueries are flattened --> for each subquery, compute similarity to all columns in all tables
    tr = [[] for _ in range(num_q)]
    for dq_idx, dq_score in enumerate(dq_scores[q_idx]):
      for t in top_tables:
        tr[dq_idx].append(dq_score[corpus_tables.index(t)])      
    
    # normalize scores
    if lm == 'tapas' and dataset == 'spider':
      score_max, score_min = np.amax(score), np.amin(score)
      score = (score - score_min) / (score_max - score_min)
    
    num_tables = len(top_tables)
    num_cols = [len(tables[t]['column_names_original']) for t in top_tables]

    cr = get_cr(dataset, top_tables)
    
    M = 1000000

    # decision variables: whether to choose a table
    b = [m.add_var(var_type=BINARY, name=f'b{i}') for i in range(num_tables)]
    
    # decision variable: which table to give the K flow
    e = [m.add_var(var_type=BINARY, name=f'e{i}') for i in range(num_tables)]
    # decision variable: flow from source to each table node
    fs = [m.add_var(var_type=CONTINUOUS, name=f'fs{i}', lb=0) for i in range(num_tables)]
    # decision variable: flow from each table node to target (sink)
    ft = [m.add_var(var_type=CONTINUOUS, name=f'ft{i}', lb=0, ub=1) for i in range(num_tables)]
    
    # decision variable: which sub-query should be covered by which column
    dr = [[[m.add_var(var_type=BINARY, name=f'dr_{i}_{j}_{k}') for k in range(num_cols[j])] for j in range(num_tables)] for i in range(num_q)]
    # decision variable: whether a question is covered or not
    q = [m.add_var(var_type=BINARY, name=f'q{i}') for i in range(num_q)]

    # decision variable: which column pair should be chosen
    c_ij_kl = [[[[0 for l in range(num_cols[j])] for k in range(num_cols[i])] for j in range(num_tables)] for i in range(num_tables)]
    # decision variable: flow for each column apir
    f_ij_kl = [[[[0 for l in range(num_cols[j])] for k in range(num_cols[i])] for j in range(num_tables)] for i in range(num_tables)]
    for i in range(num_tables):
      for j in range(num_tables):
        for k in range(num_cols[i]):
          for l in range(num_cols[j]):
            if cr[(i, j)][k][l] > 0 and j != i:
              c_ij_kl[i][j][k][l] = m.add_var(var_type=BINARY, name=f'cjoin_{i}_{j}_{k}_{l}')
              f_ij_kl[i][j][k][l] = m.add_var(var_type=CONTINUOUS, name=f'f{i}_{j}_{k}_{l}', lb=0)
    
    # Equation 2
    m += xsum(b[i] for i in range(num_tables)) == k_num
    m += xsum(c_ij_kl[i][j][k][l] for i in range(num_tables) for j in range(num_tables) for k in range(num_cols[i]) for l in range(num_cols[j])) <= join_num

    for i in range(num_tables):
      for j in range(num_tables):
        # Equation 4
        m += xsum(c_ij_kl[i][j][k][l] for k in range(num_cols[i]) for l in range(num_cols[j])) <= 1 
        
        for k in range(num_cols[i]):
          for l in range(num_cols[j]):
            # skip column pairs with compatibility = 0
            if cr[(i, j)][k][l] > 0 and j != i:
              # Equation 3
              m += 2 * (c_ij_kl[i][j][k][l] + c_ij_kl[j][i][l][k]) <= b[i] + b[j]
              # Equation 15
              m += 1/k_num * f_ij_kl[i][j][k][l] <= c_ij_kl[i][j][k][l]
              m += c_ij_kl[i][j][k][l] <= M * f_ij_kl[i][j][k][l]

    for i in range(num_q):
      for j in range(num_tables):
        # Equation 5
        m += xsum(dr[i][j][k] for k in range(num_cols[j])) <= 1
    
    for j in range(num_tables):        
      # Equation 6
      for k in range(num_cols[j]):
        m += 1/num_q * xsum(dr[i][j][k] for i in range(num_q)) <= b[j]
    
    for i in range(num_q):
      # Equation 9
      m += q[i] <= xsum(dr[i][j][k] for j in range(num_tables) for k in range(num_cols[j]))
    
    # Equation 10
    m += xsum(dr[i][j][k] for i in range(num_q) for j in range(num_tables) for k in range(num_cols[j])) <= w5 * num_q
  
    # Equation 12
    for i in range(num_tables):
      m += fs[i] + xsum(f_ij_kl[j][i][l][k] for j in range(num_tables) for l in range(num_cols[j]) for k in range(num_cols[i])) == ft[i] + xsum(f_ij_kl[i][j][k][l] for j in range(num_tables) for k in range(num_cols[i]) for l in range(num_cols[j]))

    # Equation 13
    m += xsum(ft[i] for i in range(num_tables)) == k_num

    # Euqation 14
    m += xsum(e[i] for i in range(num_tables)) == 1
    for i in range(num_tables):
      m += e[i] <= M * b[i]
      m += fs[i] == k_num * e[i]

    # objective: question-table fine-grained relevance
    obj = w1 * xsum(tr[i][j][k] * dr[i][j][k] for i in range(num_q) for j in range(num_tables) for k in range(num_cols[j]))

    # objective: sub-query coverage
    obj += w2 * xsum(q[i] for i in range(num_q))

    # objective: question-table coarse-grained relevance
    obj += w4 * xsum(score[corpus_tables.index(top_tables[i])]*b[i] for i in range(num_tables))

    # objective: table-table compatibility score
    obj += xsum(w3 * 0.5*(cr[(i, j)][k][l] + cr[(j, i)][l][k]) * c_ij_kl[i][j][k][l] for i in range(num_tables) for j in range(num_tables) for k in range(num_cols[i]) for l in range(num_cols[j]))

    m.objective = maximize(obj)

    m.verbose = 0
    status = m.optimize(max_seconds=60)
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
      r_idxs = [int(v.name[1:]) for v in m.vars if abs(v.x) > 1e-6 and 'b' in v.name and '_' not in v.name]
      r_tables = [top_tables[r_idx] for r_idx in r_idxs]
      print(r_tables)
      preds.append(r_tables)
    else:
      preds.append([])
    
    write_json(preds, fn)
  
  write_json(preds, fn)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--partition', type=int)
  args = parser.parse_args()
  num_partitions = 40

  dataset = ['bird', 'spider'][0]
  model = ['contriever', 'tapas'][0]

  q_scores = np.load(f'./data/{dataset}/{model}/score.npy')
  print(q_scores.shape)

  dq = read_json(f'./data/{dataset}/decomp.json')
  dq_scores = read_pickle(f'./data/{dataset}/contriever/score_decomp.pkl')
  q_interval = [0]
  for q in dq:
    q_interval.append(q_interval[-1] + len(q))
  dq_scores = [dq_scores[q_interval[i]:q_interval[i+1]] for i in range(len(dq))]

  # k_num is the number of tables in the output
  # t_num is the number of tables provided to the MIP program (from contriever/ tapas)
  k_num, t_num = 2, 20
  
  if dataset == 'bird':
    if model == 'contriever':
      w = [3, 8, 1, 3, 1]
    elif model == 'tapas':
      w = [3, 8, 1, 0.3, 1]
  elif dataset == 'spider':
    w = [2, 8, 1, 10, 1]

  fn = f'./data/ilp_preds/{model}/{dataset}_k_{k_num}'
  print(fn)

  preds = assign_columns(model, q_scores, dq_scores, dataset, k_num, k_num-1, t_num, w, num_partitions, args.partition, fn)
  # merge(num_partitions, fn, 'json')
  # eval_preds(dataset, read_json(f'{fn}.json'))