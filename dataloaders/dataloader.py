import torch
import random
import pandas as pd

from torch.utils.data import DataLoader


from dataloaders.common import inin_w
from utils.utils import load_module_


class Dataloader:
    def __init__(self, dataset_name, extension):
        self.dataset = self.find_definition(dataset_name, extension)

    def find_definition(self, dataset_name, extension):
        
        m = load_module_(extension, 'dataloaders', dataset_name)
        return m.__dict__['Dataset']
            

    def get_args(self, parser):
        parser.add('--num_workers', type=int, default=4,   help='Number of data loading workers.')
        parser.add('--batch_size',  type=int, default=64,  help='Batch size (train)')
        parser.add('--batch_size_val',  type=int, default=64,  help='Batch size (val)')
        parser.add('--num_samples_train',  type=int, default=-1, help='Image size')

        parser.add('--train_phase_mode',     default='train="crop,augment"^val="crop"',  action="store_bool")
        parser.add('--test_phase_mode',      default='crop',  action="store_bool")

        # parser.add('--sampler',  type=int, default=10000, help='Image size')
        # parser.add('--sampler_args',  type=int, default=10000, help='Image size')
        
        return self.dataset.get_args(parser)


    def get_dataloader(self, args, model_native_transform, part, phase, sampler=None):
        
        if hasattr(self.dataset, 'get_dataloader'):
            return self.dataset.get_dataloader(args, model_native_transform, part, phase)
        else:
            
            dataset = self.dataset.get_dataset(args, model_native_transform, part, phase)


            if args.num_samples_train != -1 and phase =='train': 
                sampler = torch.utils.data.RandomSampler(range(len(dataset)), replacement=True, num_samples=args.num_samples_train)

            if sampler is None: 
                sampler = torch.utils.data.RandomSampler(range(len(dataset)), replacement=False)

            batch_size = args.batch_size if phase == 'train' else args.batch_size_val

            return DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            sampler=sampler if phase == 'train' else None,
                            pin_memory=True,
                            drop_last=True if phase == 'train' else False,
                            shuffle=None if phase == 'train' else True,
                            worker_init_fn=inin_w)




