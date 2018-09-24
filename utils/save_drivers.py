import numpy as np
from typing import Dict

import pandas as pd


def npz_per_item(data: Dict[str, np.ndarray], path: str, args=None) -> None:
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys

    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    np.savez_compressed(path, **data)


def save_img_pil(img_np, save_path):
    Image.fromarray(img_np).save(save_path, quality=100, optimize=True, progressive=True)

class ImagesWithPostfixes(object):
    
    def get_args(self, parser):
        parser.add('--print_frequency', type=int, default=50)
        parser.add('--save_postfix',  type=int, default=0)

        return parser

    def __init__(self, arg):
        super(ImagesWithPostfixes, self).__init__()
        self.arg = arg
        
def images_with_postfixes(data: Dict[str, np.ndarray], path: str, args=None) -> None:
    name = data["name"]

    pred   = (data['pred'][:3].transpose(1, 2, 0)   * 255).astype(np.uint8)
    target = (data['target'][:3].transpose(1, 2, 0) * 255).astype(np.uint8)

    basename = os.path.basename(name)

    input_name  = f'{path}_x.png'
    target_name = f'{path}_gt.png'
    pred_name   = f'{path}_{args.save_postfix}.png'

    if not os.path.exists(x_name):
        save_img_pil((x[:3].transpose(1, 2, 0) * 255).astype(np.uint8), x_name)
    if not os.path.exists(y_name):
        save_img_pil((y[:3].transpose(1, 2, 0) * 255).astype(np.uint8), y_name)

    save_img_pil((pred, o_name)


def pandas_msg_compressed(output: Dict[str, np.ndarray], path: str, args=None):
    df = pd.DataFrame.from_dict(output, orient='index')
    df.to_msgpack(path, compress='zlib')
