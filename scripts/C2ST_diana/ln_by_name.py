import pandas as pd
import os
import argparse 

from tqdm import tqdm 

"""
    Assumes the datasets are aligned 
"""

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--img_dir',   type=str, default="", help='')
parser.add('--out_dir',   type=str, default="", help='')
parser.add('--file_list', type=str, default="", help='')

args = parser.parse_args()

files = pd.read_csv(args.file_list, header=None)[0]

for path in tqdm(files): 
    name = os.path.basename(path)
    
    os.symlink(os.path.abspath(f'{args.img_dir}/{name}'), 
               os.path.abspath(f'{args.out_dir}/{name}'))