import cv2
import numpy as np
from glob import glob
import os.path 
import argparse 
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed
import colored_traceback.auto

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--in_dir',  type=str, help='', default='')
parser.add('--in_txt',  type=str, help='', default='')
parser.add('--out_txt',  type=str, help='', default='')

parser.add('--num_workers',  type=int, default=4, help='')

args = parser.parse_args()


img_exts = ['jpg', 'png', 'jpeg', 'tiff', 'bmp']

def check(src):
    img = cv2.imread(src)

    if img is None:
        print(src)
        return False
    else:
        return True


if args.in_txt == '':
    in_paths = sum([glob(f'{args.in_dir}/*.{x}') for x in img_exts] + [glob(f'{args.in_dir}/*.{x.upper()}') for x in img_exts], [])
else:
    in_paths = np.loadtxt(args.in_txt, dtype=str)

print(f'Number of input images: {len(in_paths)}')

out = Parallel(n_jobs=args.num_workers, verbose=0)(delayed(check)(src) for src in tqdm(in_paths))


np.savetxt(args.out_txt + '_bad', [path for is_ok, path in zip(out, in_paths) if not is_ok], fmt='%s')
np.savetxt(args.out_txt + '_good', [path for is_ok, path in zip(out, in_paths) if is_ok], fmt='%s')