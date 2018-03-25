import pandas as pd
import os.path
import argparse 
import numpy as np
import json 

from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
"""
    Assumes the datasets are aligned and img names share prefix    

python scripts/attributes_classifier/train_test_split.py \
--class data/attribute_classifier/datasets/race/asian/ \
--class data/attribute_classifier/datasets/race/black \
--class data/attribute_classifier/datasets/race/white \
--save_dir data/attribute_classifier/datasets/race --same_size

"""

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--class', type=str, action='append', dest='classes', default=[], help='')
# OR
parser.add('--classes_dir', type=str, default="", help='')
parser.add('--test_dir', type=str, default="test dir without folder structure")

parser.add('--save_dir', type=str, default="", help='', required=True)

parser.add('--val_size', type=float, default=0.10)
parser.add('--test_size', type=float, default=0.10)

parser.add('--aligned', default=False, action='store_true')
parser.add('--same_size', default=False, action='store_true')
parser.add('--random_state', type=int, default=660, help='')

args = parser.parse_args()


if args.classes_dir != "":
    assert len(args.classes) == 0, 'Please pass either --class <> either --classes_dir <>'
    classes = sorted(glob(f"{args.classes_dir}/*/"))


# Get all imgs

encoding = {}
all_imgs, all_labels = [], []
for i, dir_path in enumerate(args.classes):
    imgs = [os.path.abspath(x) for x in sorted(glob(f'{dir_path}/*'))]
    labels = [i]*len(imgs)    

    print(f'Found {len(imgs)} images for class {i}.')
    all_imgs.append(imgs)
    all_labels.append(labels)

    encoding[i] = os.path.basename(os.path.abspath(dir_path))


# Balance classes

if args.same_size:
    min_size = min([len(x) for x in all_imgs])
    all_imgs = [x[:min_size] for x in all_imgs]
    all_labels = [x[:min_size] for x in all_labels]

    print(f'Reduced to equal size of {len(all_imgs[0])}.')

all_imgs, all_labels = sum(all_imgs, []), sum(all_labels, [])
df = pd.DataFrame(np.vstack([all_imgs, all_labels]).T, columns=['img_path', 'label'])


# Split

# Assume shared prefix if aligned
df['basename'] = df.img_path.apply(os.path.basename)
df = df.sort_values('basename')

df_train, df_test = train_test_split(df, random_state=args.random_state, shuffle=not args.aligned)
if not args.aligned: 
    df = shuffle(df).reset_index(drop=True)
    print (df.index)

train_size = 1.0 - args.val_size - args.test_size
df_train, df_val, df_test = np.split(df, [int(train_size*len(df)), int((1 - args.test_size)*len(df))])


# Get test images

if (args.test_size == 0) and (args.test_dir != ""):
    print(f'Generating test set from directory {args.test_dir}')
    imgs = [os.path.abspath(x) for x in sorted(glob(f'{args.test_dir}/*'))]
    labels = [0] * len(imgs)

    df_test = pd.DataFrame(np.vstack([imgs, labels]).T, columns=['img_path', 'label'])
    df_test['basename'] = df_test.img_path.apply(os.path.basename)


# Save to disk

df_train.to_csv(f'{args.save_dir}/train.csv', index=False)
df_val.to_csv  (f'{args.save_dir}/val.csv',   index=False)
df_test.to_csv (f'{args.save_dir}/test.csv',  index=False)

print(f"Encoding: {encoding}")
with open(f'{args.save_dir}/encoding.json', 'w') as fp:
    json.dump(encoding, fp)