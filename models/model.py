import torch
from models.LSUV import LSUVinit
from huepy import red, yellow
import os 

def get_model_args(get_args):
    def wrapper(parser):
        parser.add('--checkpoint', type=str, default="")
        parser.add('--net_init', type=str, default="", help='pretrained|lsuv|default')
        parser.add('--lsuv_batch_size', type=int, default=-1)
        parser.add('--use_all_gpus', default=False, action='store_bool')

        return get_args(parser)

    return wrapper


def get_abstract_net(get_net):
    def wrapper(args, train_dataloader):

        model = get_net(args)
        model = model.to(args.device)


        if args.use_all_gpus and args.device == 'cuda':
            print(yellow('Using all GPU\'s!'))
            model = torch.nn.DataParallel(model)

        if args.net_init == 'lsuv' and args.checkpoint == '':
            data = iter(dataloader_train).next()[1].to(args.device)

            if args.lsuv_batch_size > 0:
                data = data[:args.lsuv_batch_size]
            model = LSUVinit(model, data, needed_std=1.0, std_tol=0.1,
                             max_attempts=10, do_orthonorm=False, device=args.device)
            
        if args.checkpoint != '':

            if os.path.exists(args.checkpoint):
                checkpoint_path = args.checkpoint
            elif args.experiment_dir != '' and os.path.exists(os.path.join(args.experiment_dir, 'checkpoints', args.checkpoint)):
                checkpoint_path = os.path.join(args.experiment_dir, 'checkpoints', args.checkpoint)
            else:
                print(args.checkpoint)
                print(args.experiment_dir)
                assert False

            print(f"Loading {red(args.model)} from checkpoint {red(checkpoint_path)}")
            model.load_state_dict(torch.load(checkpoint_path))

        return model

    return wrapper
