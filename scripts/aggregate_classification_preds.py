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
import torch
from pathlib import Path
"""
    Template


python scripts/preds_to_submission.py \
--encoding extensions/superheroes/data/encoding.json \
--preds_path extensions/superheroes/data/preds.pickle \
--save_path p.csv


"""

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument


parser.add('--preds_dir', type=Path)

args = parser.parse_args()


files = glob(f'{args.preds_dir}/*.npz')

preds = []
names = []
labels = []
for i, p in enumerate(tqdm(files)):

    if p.endswith('all.npz'):
        continue
        
    loaded = np.load(p)
    pred = loaded['output'][0]
          

    names.append(np.vstack(loaded['names']))
    preds.append(np.vstack(pred))
    labels.append(np.hstack(loaded['labels']))
# 

names  = np.vstack(names)[:, 0]
labels = np.hstack(labels)
preds_= torch.softmax(torch.from_numpy(np.vstack(preds)), 1).numpy()

np.savez_compressed(args.preds_dir / 'all', preds=preds_, names=names, labels=labels)  
