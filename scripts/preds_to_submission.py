import pandas as pd
import os.path
import argparse 
import numpy as np
import json
import pickle

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
"""
    Assumes the datasets are aligned and img names share prefix    
"""

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--encoding', type=str,    default="", help='')

parser.add('--preds_path', type=str, default="", help='')

parser.add('--save_path', type=str, default="", help='')


args = parser.parse_args()

with open(f'{args.encoding}', 'r') as fp:
    encoding = json.load(fp)

with open(f'{args.preds_path}', 'rb') as fp:
    d = pickle.load(fp)

print(encoding)

preds = []
for path, preds_obj in d.items():    
    preds.append([np.argmax(c) for c in preds_obj])        

df = pd.DataFrame(preds, columns=['Superhero'])
df.Superhero = df.Superhero.astype(str).map(encoding)
df['filename'] = [os.path.basename(x)[:-4] for x in d.keys()] 

df['rr'] = df.filename.apply(lambda x: int(x.split('_')[-1]))
df = df.sort_values('rr')

df = df.drop('rr', axis=1)


df[['filename', 'Superhero']].to_csv(f"{args.save_path}", index=False)