import json
import junky
from mordl import NeTagger
import re
import sys
BERT_MODEL_FN = 'bert-base-multilingual-cased'
MODEL_FN = 'misc-ne-bert_model'
SEED=42
BERT_MAX_LEN, BERT_EPOCHS, BERT_BATCH_SIZE = 512, 10, 8

def model(input=None):
  res_dev =  MODEL_FN.replace('_model', '.conllu')
  res_test = MODEL_FN.replace('_model', '.conllu')
  tagger = NeTagger()
  tagger.load(MODEL_FN)
  junky.clear_tqdm()
  file = next(tagger.predict(input[1], clone_ds=True))
  j_son = {'q1': [],
           'q2': [],
           'q3': [],
           'q4': [],
           'q5': [],
           'q6': [],
           'q7': [],
           'q8': [],
           'q9': [],
           'q10': []}
  for i in file[0]:
    for j in j_son.keys():
      if len(i['MISC'].keys()) > 0:
        if i['MISC']['NE'] == j.upper():
          j_son[j].append(i['FORM'])
  j_son['q1'] = " ".join(j_son['q1'])
  a = None if re.sub(r'[^0-9]', '', " ".join(j_son['q1'])) == '' else re.sub(r'[^0-9]', '', " ".join(j_son['q1']))
  b = None if re.sub(r'[^0-9]', '', " ".join(j_son['q4'])) == '' else re.sub(r'[^0-9]', '', " ".join(j_son['q4']))
  c = None if " ".join(re.sub(r'[^0-9]', ' ', " ".join(j_son['q10'])).split()) == '' else [int(i) for i in
                                                                                           re.sub(r'[^0-9]', ' ',
                                                                                                  " ".join(j_son[
                                                                                                             'q10'])).split()]
  if a == None:
    j_son.pop('q2')
  else:
    j_son['q2'] = int(a)
  if b == None:
    j_son.pop('q4')
  else:
    j_son['q4'] = int(b)

  if c == None:
    j_son.pop('q10')
  else:
    j_son['q10'] = c
  j_son['q5'] = " ".join(j_son['q5'])
  j_son['q3'] = " ".join(j_son['q3'])
  return j_son

print(model(sys.argv))