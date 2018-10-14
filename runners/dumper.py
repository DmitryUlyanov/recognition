import torch

import typed_print as tp
from huepy import red, lightblue, orange
print = tp.init(palette='dark', # or 'dark' 
                str_mode=tp.HIGHLIGHT_NUMBERS) 
                # highlight_word_list=['Training'])

def get_args(parser):
    parser.add('--num_batches',  default=1, type=int)
    parser.add('--dump_save_dir',  default='dumper', type=str)
    return parser


def run_epoch(dataloader, model, criterion, optimizer, epoch, args, part='train'):
    for it, data in enumerate(dataloader):
        if it >= args.num_batches:
          break

    print(f'Dumping {part}', it)
    torch.save(data, f'{args.dump_save_dir}/data_{part}_{it}.pth')

    return 0
