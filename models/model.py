import torch
from models.LSUV import LSUVinit


def get_model_args(get_args):
    def wrapper(parser):
        parser.add('--checkpoint', type=str, default="")
        parser.add('--net_init', type=str, default="")
        parser.add('--lsuv_batch_size', type=int, default=-1)
        parser.add('--use_all_gpus', action='store_true')

        return get_args(parser)

    return wrapper


def get_abstract_net(get_net):
    def wrapper(args, train_dataloader):

        model, criterion = get_net(args)
        model, criterion = model.to(args.device), criterion.to(args.device)


        if args.use_all_gpus and args.device == 'gpu':
            model = torch.nn.DataParallel(model)

        if args.net_init == 'lsuv' and args.checkpoint == '':
            data = iter(dataloader_train).next()[1].to(args.device)

            if args.lsuv_batch_size > 0:
                data = data[:args.lsuv_batch_size]
            model = LSUVinit(model, data, needed_std=1.0, std_tol=0.1,
                             max_attempts=10, do_orthonorm=False, device=args.device)
            
        if args.checkpoint != '':
            print(f"Loading pretrained net: {args.model} from {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint))

        return model, criterion

    return wrapper
