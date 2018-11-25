import pandas as pd
import os.path
import argparse 
import numpy as np
import json
import pickle
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
"""
    Template


python scripts/preds_to_submission.py \
--encoding extensions/superheroes/data/encoding.json \
--preds_path extensions/superheroes/data/preds.pickle \
--save_path p.csv


"""

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--encoding', type=str,    default="", help='')
parser.add('--preds_dir', type=str, default="", help='')
parser.add('--save_path', type=str, default="", help='')
parser.add('--sample_submission', type=str, default="", help='')

args = parser.parse_args()


with open(f'{args.encoding}', 'r') as fp:
    encoding = json.load(fp)
mp = {i: v for i, v in enumerate(encoding)}
print(mp)

sample_submission = pd.read_csv(args.sample_submission)



# preds_dirs = ['../data/experiments/11-22,13:24;config_name:resnet101_stage3']

files = glob(f'{args.preds_dir}/*.npz')

preds = []
for i, p in enumerate(tqdm(files)):
    loaded = np.load(p)
    pred = loaded['output'][0]
    pred_classes = np.argsort(pred, 1)[:, -3:][:, ::-1]
    preds.append(np.hstack([ loaded['names'][:, None], pred_classes] ))
    
    
ans = pd.DataFrame(np.vstack(preds), columns=['key_id', 1, 2, 3])
# print(ans)
ans['word'] =  ans[1].map(mp).str.replace(' ', '_') + ' ' + ans[2].map(mp).str.replace(' ', '_') + ' ' + ans[3].map(mp).str.replace(' ', '_')
print(ans['word'])
# word.replace(' ', '_')
ans = ans.drop([1, 2, 3], axis=1)
ans.to_csv(args.save_path + '.gz', compression='gzip', index=False)