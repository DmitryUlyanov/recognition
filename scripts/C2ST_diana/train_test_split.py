import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import os.path
import argparse 
import numpy as np
"""
    Assumes the datasets are aligned and img names share prefix    
"""

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--dir_class0', type=str, default="", help='')
parser.add('--dir_class1', type=str, default="", help='')

parser.add('--random_state', type=int, default=660, help='')

parser.add('--save_dir', type=str, default="", help='')

parser.add('--aligned', default=False, action='store_true')
parser.add('--test_size', type=float, default=0.15)
parser.add('--same_size', default=False, action='store_true')

args = parser.parse_args()

# Get all imgs
all_imgs, all_labels = [], []
for i, dir_path in enumerate([args.dir_class0, args.dir_class1]):
    imgs = [os.path.abspath(x) for x in sorted(glob(f'{dir_path}/*'))]
    labels = [i]*len(imgs)    

    print(f'Found {len(imgs)} images for class {i}.')
    all_imgs.append(imgs)
    all_labels.append(labels)

if args.same_size:
    min_size = min([len(x) for x in all_imgs])
    all_imgs = [x[:min_size] for x in all_imgs]
    all_labels = [x[:min_size] for x in all_labels]

    print(f'Reduced to equal size of {len(all_imgs[0])}.')

all_imgs, all_labels = sum(all_imgs, []), sum(all_labels, [])

df = pd.DataFrame(np.vstack([all_imgs, all_labels]).T, columns=['img_path', 'label'])

# Assume shared prefix
df['basename'] = df.img_path.apply(os.path.basename)
df = df.sort_values('basename')

df_train, df_test = train_test_split(df, random_state=args.random_state, shuffle=not args.aligned)

df_train.to_csv(f'{args.save_dir}/train.csv', index=False)
df_test.to_csv (f'{args.save_dir}/test.csv', index=False)