import numpy as np
from typing import Dict

import pandas as pd


def npz_per_item(output: Dict[str, np.ndarray], path: str) -> None:
    """
    Saves predictions to npz format, using one npy per sample,
    and sample names as keys

    :param output: Predictions by sample names
    :param path: Path to resulting npz
    """

    np.savez_compressed(path, **output)


def pandas_msg_compressed(output: Dict[str, np.ndarray], path: str):
    df = pd.DataFrame.from_dict(output, orient='index')
    df.to_msgpack(path, compress='zlib')
