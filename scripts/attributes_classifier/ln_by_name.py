import pandas as pd
import os
import argparse 

from glob import glob
from tqdm import tqdm 

"""
    
"""

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--img_dir',   type=str, default="", help='')
parser.add('--out_dir',   type=str, default="", help='')
parser.add('--file_list', type=str, default="", help='')

parser.add('--others', default=False, action='store_true')

args = parser.parse_args()

files = pd.read_csv(args.file_list, header=None)[0]

if args.others:
    files_all = [os.path.basename(x) for x in os.glob(f'{img_dir}/*')]
    files = list( set(files_all) - set(files) )

print(len(files))

for path in tqdm(files): 
    name = os.path.basename(path)
    
    os.symlink(os.path.abspath(f'{args.img_dir}/{name}'), 
               os.path.abspath(f'{args.out_dir}/{name}'))