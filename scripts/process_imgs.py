from PIL import Image
import shutil
from skimage import color
import numpy as np
import argparse 
import cv2 

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

# parser.add('--img_list',  type=str, help='', default='')

parser.add('--in_dir',  type=str, help='', default='')
parser.add('--out_dir',  type=str, default='', help='')



args = parser.parse_args()


# def get_sin(x, weights):
#     # weighted = [w * x[:,:, i] for w, i in zip(weights, range(3))]
    	
#     return ((np.sin(np.sum(x.astype(np.float32) /255. * weights[None, None, :], 0))[:, :, None]+ 1) /2 * 255).astype(np.uint8)

    # return np.sin(np.sum(x * weights[None, None, :], 0))[:, :, None]


weights = np.random.randn(20, 3) * np.arange(1, 21)[:, None] * 4
print(weights)

def copy_resize(src, dst):
    if '.png' in src:
        # print (sfrfrc)
        img = cv2.imread(src, -1)

        sins = [] 
        for i in range(20):
        	
        	q = img.astype(np.float32) / 255.
        	q1 = np.sum(q * weights[i][None, None, :], 2)

        	q2 = (np.sin(q1) + 1) / 2. * 255

        	sins.append(q2.astype(np.uint8))


        # color.colorconv.lab_ref_white = np.array([0.96422, 1.0, 0.82521])
        # img_lab = color.rgb2lab(np.array(img))
        np.savez_compressed(dst + '.npz', np.array(sins))

    # else:
    #     shutil.copy2(src, dst)


# src_path = 'data/test'
# dst_path = 'data/raw/256/test'

# src_path = 'data/train'
# dst_path = 'data/raw/256/train'

# src_path = 'data/raw/256/test'
# dst_path = 'data/raw/256_lab/test'

shutil.copytree(args.in_dir, args.out_dir, copy_function=copy_resize)

