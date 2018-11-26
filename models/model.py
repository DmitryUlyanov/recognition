import torch
from models.LSUV import LSUVinit
from huepy import red, yellow
from utils.utils import load_module
from munch import munchify
from models.common import set_param_grad
from dataloaders.augmenters import Identity
import os 

def get_model_args(get_args):
    def wrapper(parser):
        parser.add('--checkpoint', type=str, default="")
        parser.add('--net_init', type=str, default="", help='pretrained|lsuv|default')
        parser.add('--lsuv_batch_size', type=int, default=-1)
        parser.add('--use_all_gpus', default=False, action='store_bool')
        parser.add('--parallel_criterion', default=False, action='store_bool')
        parser.add('--fix_feature_extractor', default=False, action='store_bool')
        parser.add('--freeze_bn', default=False, action='store_bool')
        parser.add('--merge_model_and_loss', default=False, action='store_bool')

        return get_args(parser)

    return wrapper


def get_abstract_net(get_net):
    def wrapper(args, train_dataloader, criterion):

        model = get_net(args)
        model = model.to(args.device)


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

            print(f" - Loading {yellow(args.model)} from checkpoint {yellow(checkpoint_path)}")
            
            state_dict = torch.load(checkpoint_path)
            if isinstance(state_dict, dict): 
                state_dict = state_dict['state_dict']
            
            model.load_state_dict(state_dict)
        
        
        if hasattr(model, 'feature_extractor'):
            value = not args.fix_feature_extractor
            set_param_grad(model.feature_extractor, value = value, set_eval_mode = False)

        if args.freeze_bn:
            print(yellow(' - Freezing BN'))

            def freeze_bn(m):
                if isinstance(m, torch.nn.BatchNorm2d):

                    m.training = False
                    
                    def nop(*args, **kwargs):
                        pass

                    m.train = nop

            model.apply(freeze_bn)
        
        if args.merge_model_and_loss:
            model = ModelAndLoss(model, criterion)

        if args.use_all_gpus and args.device == 'cuda' and torch.cuda.device_count() > 1:
            print(yellow(' - Using all GPU\'s!'))

            if args.parallel_criterion:
                import encoding
                model = encoding.parallel.DataParallelModel(model)       
            else:
                model = torch.nn.DataParallel(model) 
            # encoding.parallel.patch_replication_callback(model)
            # import encoding
            # encoding.parallel.DataParallel(model)



        return model

    return wrapper

def get_abstract_native_transform(get_native_transform):
    def wrapper():
        
        res = get_native_transform()
        if res is None:
            res = Identity

        return res

    return wrapper

def save_model(model, epoch, args, optimizer=None):
    
    model_to_save = model
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module

    if 'ModelAndLoss' in str(type(model_to_save)):
        model_to_save = model_to_save.model

    dict_to_save = { 
        'state_dict': model_to_save.state_dict(), 
        'args': args,
        'optimizer': optimizer.state_dict() if optimizer is not None else None
    }
    save_path = f'{args.experiment_dir}/checkpoints/model_{epoch}.pth'

    torch.save(dict_to_save, save_path, pickle_protocol=-1)


class BaseModel(torch.nn.Module):
    def __init__(self, feature_extractor, predictor):
        super(BaseModel, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.predictor = predictor
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        x = self.feature_extractor(input)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.predictor(x)

        return x
        


def load_model_from_checkpoint(checkpoint_path, args_to_update=None):

        args = vars(torch.load(checkpoint_path)['args'])

        args['checkpoint'] = checkpoint_path
        args['net_init'] = 'checkpoint'
        args['use_all_gpus'] = False
        args['parallel_criterion'] = False
        args['freeze_bn'] = False

        if checkpoint_path == 'extensions/rawr/data/lrnet/experiments/10-12,13:20;config_name:lrnet/checkpoints/model_50.pth':
            args['predictor_config']['return_features'] = False
            args['processor_config']['num_maps_input'] = 3
            args['processor_config']['filter_size'] = 3
            args['processor_config']['model'] += '_'
        
        if 'processor_config' in args:    
            args['processor_config']['freeze_bn'] = False
        
        if 'predictor_config' in args:
            args['predictor_config']['freeze_bn'] = False

        if args_to_update is not None:
            args.update(args_to_update)

        m_model = load_module(args['extension'], 'models', args['model'])

        print(args)
        model = m_model.get_net(munchify(args), None)

        return model, args




class ModelAndLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelAndLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, target, *data):
        output = self.model(*data)
        loss = self.loss(output, target)

        return loss, output












