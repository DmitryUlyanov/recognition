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


def run_epoch_train(dataloader, model, criterion, optimizer, epoch, args):
  for it, data in enumerate(dataloader):
    if it >= args.num_batches:
      break

    print('Dumping train', it)
    torch.save(data, f'{args.dump_save_dir}/data_train_{it}.pth')
    
        
def run_epoch_test(dataloader, model, criterion, epoch, args, need_softmax=False, need_preds=False):
  for it, data in enumerate(dataloader):
    
    if it >= args.num_batches:
      break
    
    print('Dumping test', it)
    torch.save(data, f'{args.dump_save_dir}/data_test_{it}.pth')
    

  return -1

