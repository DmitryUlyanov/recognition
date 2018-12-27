import cv2
import numpy as np
from glob import glob
import os.path 
import argparse 
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed
import colored_traceback.auto
from huepy import red 

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--img_list',  type=str, help='', default='')
# then saved to a path/resized/
# OR
parser.add('--in_dir',  type=str, help='', default='')
parser.add('--out_dir',  type=str, default='', help='')


parser.add('--out_ext',    type=str, default='png', help='')
parser.add('--out_jpg_q',  type=int, default=99,  help='')

parser.add('--out_img_width',   type=int, default=0,  help='')
parser.add('--out_img_height',  type=int, default=0,  help='')

parser.add('--save_aspect_ratio',  action='store_true')
parser.add('--max_dim',  type=int, default=0,  help='')


parser.add('--overwrite',  action='store_true')

parser.add('--num_workers',  type=int, default=4, help='')


parser.add('--remove_alpha',  action='store_true')

args = parser.parse_args()


def image_resize(image, specific_size = None, max_dim = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    assert not (specific_size is None and max_dim is None)

    if specific_size is not None:
        height, width = specific_size
    else:

        max_dim_ = float(max(h, w))

        height, width = int(round(h / max_dim_ * max_dim)), int(round(w / max_dim_ * max_dim))  
        
        # print(height, width)

    resized = cv2.resize(image, (width, height), interpolation = inter)

    return resized



def convert(src, dst, args):
    if args.overwrite or not os.path.exists(dst):

        img = cv2.imread(src, -1)
        if img is None:
            print(red(src))
            return
        
        if (args.out_img_height > 0) and (args.out_img_width > 0) or args.max_dim > 0 :
            if args.save_aspect_ratio:
                img = image_resize(img, max_dim=args.max_dim)
            else:
                img = image_resize(img, specific_size=(args.out_img_height, args.out_img_width)) 

        if args.remove_alpha and len(img.shape):
            img = img[:, :, :3]

        if args.out_ext == 'png':
            cv2.imwrite(dst, img,  [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elif args.out_ext == 'jpg':
            cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), args.out_jpg_q])            
    
def basename_no_ext(x):
    return '.'.join(os.path.basename(x).split('.')[:-1])



if args.in_dir != '':
    img_exts = ['jpg', 'png', 'jpeg', 'tiff', 'bmp']
    in_paths = sum([glob(f'{args.in_dir}/*.{x}') for x in img_exts] + [glob(f'{args.in_dir}/*.{x.upper()}') for x in img_exts], [])
    out_paths = [f"{args.out_dir}/{os.path.basename(x) + f'.{args.out_ext}'}" for x in in_paths]

    os.makedirs(args.out_dir, exist_ok = True)
elif args.img_list != '':
    in_paths = np.loadtxt(args.img_list, dtype=str)
    out_paths = [f"{os.path.dirname(x)}/resized/{os.path.basename(x) + f'.{args.out_ext}'}" for x in in_paths]
else:
    assert False

print(f'Number of input images: {len(in_paths)}')

df = Parallel(n_jobs=args.num_workers, verbose=0)(delayed(convert)(src, dst, args) for src, dst in (zip(tqdm(in_paths), out_paths)))