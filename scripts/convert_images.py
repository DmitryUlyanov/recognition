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
parser.add('--out_dir',  type=str, default='', help='')

parser.add('--out_ext',    type=str, default='', help='')
parser.add('--out_jpg_q',  type=int, default=99,  help='')

parser.add('--out_img_width',   type=int, default=0,  help='')
parser.add('--out_img_height',  type=int, default=0,  help='')

parser.add('--overwrite',  action='store_true')

parser.add('--num_workers',  type=int, default=4, help='')




args = parser.parse_args()


img_exts = ['jpg', 'png', 'jpeg', 'tiff', 'bmp']

def convert(src, dst, args):
    if args.overwrite or not os.path.exists(dst):

        img = cv2.imread(src, -1)

        if args.out_img_width > 0 and args.out_img_height > 0:
            img = cv2.resize(img, (args.out_img_height, args.out_img_width), cv2.INTER_CUBIC) 

        cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.out_jpg_q])
    
def basename_no_ext(x):
    return '.'.join(os.path.basename(x).split('.')[:-1])

os.makedirs(args.out_dir, exist_ok = True)

in_paths = sum([glob(f'{args.in_dir}/*.{x}') for x in img_exts] + [glob(f'{args.in_dir}/*.{x.upper()}') for x in img_exts], [])
out_paths = [f"{args.out_dir}/{basename_no_ext(x) + f'.{args.out_ext}'}" for x in in_paths]

print(f'Number of input images: {len(in_paths)}')

df = Parallel(n_jobs=args.num_workers, verbose=0)(delayed(convert)(src, dst, args) for src, dst in (zip(tqdm(in_paths), out_paths)))