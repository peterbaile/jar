import numpy as np

from utils import read_json, sql_to_tables

def get_p_r_f1(true_positives, false_positives, false_negatives):
  precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
  recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
  f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
  return np.array([precision, recall, f1_score])

class Metrics():
  def __init__(self, top_k = None):
    super().__init__()
    self.tp, self.fp, self.fn, self.acc, self.perfect_recall = 0.0, 0.0, 0.0, [], []
    self.p_r_f1 = np.array([0.0, 0.0, 0.0])
    self.top_k = top_k
  
  def update(self, preds: list[list[str]], target: list[list[str]]):
    cnt, pred_num, tgt_num = 0, 0, 0

    assert len(preds) == len(target)
    for pred, tgt in zip(preds, target):
      pred, tgt = [x.upper() for x in pred], [x.upper() for x in tgt]
      cnt += 1
      pred, tgt = set(pred), set(tgt)

      # treat everything as one block
      self.tp += len(pred & tgt)
      self.fp += len(pred - tgt)
      self.fn += len(tgt - pred)

      # compute independently
      _tp = len(pred & tgt)
      _fp = len(pred - tgt)
      _fn = len(tgt - pred)
      self.p_r_f1 += get_p_r_f1(_tp, _fp, _fn)

      self.acc.append(int(pred == tgt))
      self.perfect_recall.append(int(pred.issuperset(tgt)))

      pred_num += len(pred)
      tgt_num += len(tgt)
    
    print(f'average number of predicted objects: {(pred_num/cnt):.3f}')
    print(f'average number of gold objects: {(tgt_num/cnt):.3f}')
    print(f'#parsable answer: {cnt}')

    # print(f'score (one unit): {self.precision():.1f}, {self.recall():.1f}, {self.f1():.1f}, {self.accuracy():.1f}')
    self.p_r_f1 = np.round(100 * self.p_r_f1 / cnt, 1)
    self.acc = 100*np.array(self.acc).mean()
    self.perfect_recall = 100*np.array(self.perfect_recall).mean()
    print(f'score (average): {self.p_r_f1}, {self.acc:.1f}, {self.perfect_recall:.1f}')

  
  def precision(self):
    denominator = self.tp + self.fp
    if denominator == 0:
      return 0
    else:
      return self.tp / (self.tp + self.fp) * 100

  def recall(self):
    denominator = self.tp + self.fn
    if denominator == 0:
      return 0
    else:
      return self.tp / (self.tp + self.fn) * 100
  
  def f1(self):
    denominator = 2 * self.tp + self.fp + self.fn
    if denominator == 0:
      return 0
    else:
      return 2 * self.tp / (2 * self.tp + self.fp + self.fn) * 100

def eval_preds(dataset, preds):
  qs = read_json(f'./data/{dataset}/dev.json')
  golds = [sql_to_tables(q['sql'], q['db_id']) for q in qs]

  # skip questions that only involve 1 table
  preds_tmp, golds_tmp = [], []
  for pred, gold in zip(preds, golds):
    if len(gold) == 1:
      continue
    
    preds_tmp.append(pred)
    golds_tmp.append(gold)
  preds, golds = preds_tmp, golds_tmp

  score = Metrics()
  score.update(preds, golds)