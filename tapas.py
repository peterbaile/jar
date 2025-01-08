import torch
from tqdm import tqdm
import numpy as np
from torch.utils import data
import pandas as pd
import argparse
import json
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

from utils import get_corpus, read_json, merge
from metrics import eval_preds

def get_tapas_instances(dataset, corpus_tables):
  # by merging tables.json with test_t.csv
  t_df = pd.read_csv(f'./data/{dataset}/tapas/test_t.csv')
  df_table_names = t_df['schema'].tolist()
  df_table_names = ['#sep#'.join(x.split(',')[:2]) for x in df_table_names]

  instances = [t_df.iloc[df_table_names.index(t)]['instance_tapas'] for t in corpus_tables]
  return instances

class TAPASScore(nn.Module):
  def __init__(self, lm, dropout=0.2, device='cuda'):
    super(TAPASScore, self).__init__()
    print(f'model lm: {lm}')
    self.tapas = AutoModel.from_pretrained(f'google/{lm}')
    self.dropout = nn.Dropout(dropout)
    self.device = device
    hidden_size = self.tapas.config.hidden_size
    self.linear1 = nn.Linear(hidden_size, 256)
    self.linear2 = nn.Linear(256, 1)
  
  def forward(self, x):
    output = self.dropout(self.tapas(**x).pooler_output)
    output = self.dropout(self.linear1(output))
    output = self.linear2(output)
    return output

class TAPASTestDataset(data.Dataset):
  def __init__(self, questions, tables, lm, device='cuda'):
    self.ts, self.qs, self.device = tables, questions, device
    self.tokenizer = AutoTokenizer.from_pretrained(f'google/{lm}')

  def __getitem__(self, index):
    t = self.ts[index]
    t = pd.DataFrame.from_dict(json.loads(t)).astype(str)
    tokenize_output = self.tokenizer(table=t, queries=self.qs, padding=True, truncation=True, return_tensors='pt')
    return tokenize_output

  def __len__(self):
    return len(self.ts)

def get_sim_scores(qs, ts, dataset, num_partitions, partition):
  interval = len(qs) // num_partitions + 1
  start_idx, end_idx = partition * interval, (partition + 1) * interval
  qs = qs[start_idx : end_idx]

  fn = f'./data/{dataset}/tapas/score_{partition}.npy'
  
  # https://huggingface.co/docs/transformers/en/model_doc/tapas#transformers.TapasModel
  test_dataset = TAPASTestDataset(qs, ts, 'tapas-large')
  model = TAPASScore('tapas-large')
  model.load_state_dict(torch.load(f'./data/{dataset}/tapas/checkpoint.pt'))
  model = model.to('cuda')
  test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda data:data)

  scores = []
  
  model.eval()
  with torch.no_grad():
    for test_input in tqdm(test_dataloader):
      assert(len(test_input) == 1)
      test_input = test_input[0].to('cuda')
      output = model(test_input) # row is question, column is table
      scores.append(output)

  scores = torch.hstack(scores).cpu().numpy()
  np.save(fn, scores)

def evaluate(dataset, model, k):
  corpus_tables = get_corpus(dataset)
  scores = torch.from_numpy(np.load(f'./data/{dataset}/{model}/score.npy'))
  print(scores.shape)
  top_idxs = scores.topk(k=k, dim=-1)[1]

  preds = []

  for top_idx in top_idxs:
    preds.append([corpus_tables[i] for i in top_idx])
  
  eval_preds(dataset, preds)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--partition', type=int)
  args = parser.parse_args()

  dataset = ['bird', 'spider'][1]
  model = 'tapas'
  num_partitions = 8

  qs = read_json(f'./data/{dataset}/dev.json')
  qs = [q['question'] for q in qs]

  corpus_tables = get_corpus(dataset)
  tapas_instances = get_tapas_instances(dataset, corpus_tables)

  get_sim_scores(qs, tapas_instances, dataset, num_partitions, args.partition)
  # merge(num_partitions, f'./data/{dataset}/{model}/score', 'npy')
  # evaluate(dataset, model, 2)