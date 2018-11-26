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
parser.add('--preds_dir', action='append',  help='')
parser.add('--weights', type=str,  help='')
parser.add('--save_path', type=str, default="", help='')
parser.add('--sample_submission', type=str, default="", help='')

args = parser.parse_args()


with open(f'{args.encoding}', 'r') as fp:
    encoding = np.array(json.load(fp))


classes_indx = np.argsort(encoding)
encoding = encoding[classes_indx]

mp = {i: encoding[i] for i, v in enumerate(classes_indx)}
# mp = {i: v for i, v in enumerate(encoding)}

label_map = np.array([np.where(classes_indx == k)[0] for k in range(len (classes_indx)) ])[:, 0]



# print(mp)

sample_submission = pd.read_csv(args.sample_submission)


preds__ = []
for pdd in args.preds_dir:

    files = glob(f'{pdd}/*.npz')

    preds = []
    names = []
    labels = []
    for i, p in enumerate(tqdm(files)):
        loaded = np.load(p)
        pred = loaded['output'][0]
              
        names.append(np.vstack(loaded['names']))
        preds.append(np.vstack(pred))

        # print(np.hstack(loaded['y']).shape, label_map[np.hstack(loaded['y'])].shape, label_map.shape)
        labels.append(label_map[np.hstack(loaded['y'])])


    names  = np.vstack(names)[:,0]
    idxs   = np.argsort(names)
    labels = np.hstack(labels)[idxs]
    # print(labels.shape)
    names = names[idxs]
    

    preds_= torch.softmax(torch.from_numpy(np.vstack(preds)), 1).numpy()[idxs][:, classes_indx]
    # print(preds_.shape)
    preds__.append(preds_)
    # preds__.append(names)
    # preds.shape

def map3(input, target, topk=3):
    _, predictions = input.topk(topk, dim=1, sorted=True)

    apk_v = torch.eq(target, predictions[:, 0]).float()
    for k in range(1, topk):
        apk_v += torch.eq(target, predictions[:, k]).float() / (k + 1)

    return apk_v.mean()


for i, x in enumerate(preds__):
    print(i, map3(torch.from_numpy(x), torch.from_numpy(labels)).item())

print( [float(x) for x in args.weights.split(',')] )

ensemble = np.sum( [x * float(w) for x,w in zip(preds__, args.weights.split(','))],  0)


print('ensemble:', map3(torch.from_numpy(ensemble), torch.from_numpy(labels)).item())
# print(labels.shape, names.shape)    


np.savez_compressed(args.save_path + '.npz', preds=ensemble, names=names, classes=mp)  
ans = pd.DataFrame(np.argsort(ensemble, 1)[:, -3:][:, ::-1], columns=[1, 2, 3])
ans['key_id'] = names

# # print(ans)
ans['word'] =  ans[1].map(mp).str.replace(' ', '_') + ' ' + ans[2].map(mp).str.replace(' ', '_') + ' ' + ans[3].map(mp).str.replace(' ', '_')
# print(ans['word'])
# word.replace(' ', '_')
ans = ans.drop([1, 2, 3], axis=1)
ans.to_csv(args.save_path + '.gz', compression='gzip', index=False)