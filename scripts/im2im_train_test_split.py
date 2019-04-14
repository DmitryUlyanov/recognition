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

parser.add('--img_dir', type=str, action='append', dest='img_dirs', default=[], help='')
# OR
parser.add('--imgs_dir',  type=str,  default="",  help='')
parser.add('--suffix',    type=str,  default="", )
parser.add('--separator', type=str,  default="_")

parser.add('--save_dir', type=str, default="", help='', required=True)

parser.add('--val_size', type=float, default=0.10)
parser.add('--test_size', type=float, default=0.10)

# parser.add('--aligned', default=False, action='store_true')
# parser.add('--same_size', default=False, action='store_true')
parser.add('--random_state', type=int, default=660, help='')

args = parser.parse_args()

def fname(x):
    return '.'.join(os.path.basename(x).split('.')[:-1])

if args.imgs_dir != "":
    assert len(args.img_dirs) == 0, 'Please pass either --img_dir <> either --imgs_dir <>'
    # classes = sorted(glob(f"{args.imgs_dir}/*"))

else: 
    assert len(args.img_dirs) > 0, 'Please pass either --img_dir <> either --imgs_dir <>'

    dfs = []
    names = None
    for i, img_dir in enumerate(args.img_dirs):
        img_paths = glob(f'{img_dir}/*')

        colname = os.path.basename(os.path.realpath(img_dir))
        print(colname)
        df = pd.DataFrame(img_paths, columns=[colname])
        df['name'] = df[colname].apply(fname)

        dfs.append(df)        

        
    df = dfs[0]
    for d in dfs[1:]:
        df = df.join(d.set_index('name'), how='inner', on='name')

    df = df.drop('name', axis=1)

    print(f"Found {len(df)} pairs.")


df = shuffle(df).reset_index(drop=True)

train_size = 1.0 - args.val_size - args.test_size
df_train, df_val, df_test = np.split(df, [int(train_size*len(df)), int((1 - args.test_size)*len(df))])


# # Get test images

# if (args.test_size == 0) and (args.test_dir != ""):
#     print(f'Generating test set from directory {args.test_dir}')
#     imgs = [os.path.abspath(x) for x in sorted(glob(f'{args.test_dir}/*'))]
#     labels = [0] * len(imgs)

#     df_test = pd.DataFrame(np.vstack([imgs, labels]).T, columns=['img_path', 'label'])
#     df_test['basename'] = df_test.img_path.apply(os.path.basename)


# Save to disk

df_train.to_csv(f'{args.save_dir}/train.csv', index=False)
df_val.to_csv  (f'{args.save_dir}/val.csv',   index=False)
df_test.to_csv (f'{args.save_dir}/test.csv',  index=False)

# print(f"Encoding: {encoding}")
# with open(f'{args.save_dir}/encoding.json', 'w') as fp:
#     json.dump(encoding, fp)