from dataloader import *
from utils import test, setup, setup_dataset, load_model
import argparse
import importlib
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add = parser.add_argument

parser.add('--splits_dir', default='split')
parser.add('--num_workers', type=int,
           help='number of data loading workers', default=6)
parser.add('--batch_size', type=int,
           default=64, help='batch size')

parser.add('--mode', type=str, default="regression",
           help='classification|regression|regression_masked')
parser.add('--manual_seed', type=int, default=123, help='manual seed')

parser.add('--preprocess_target', type=str, default="whiten", help='none|whiten')

parser.add('--checkpoint', type=str, default="", help='')

parser.add('--model', type=str, default="", help='')

parser.add('--experiments_dir', type=str, default="experiments", help='')

# Add model args
args_, _ = parser.parse_known_args()
m = importlib.import_module('models.' + args_.model)
m.get_args(parser)


args, default_args = parser.parse_args(), parser.parse_args([])
args.phase, default_args.phase = "test", "test"


setup(args, default_args)

model, criterion = load_model(args)

use_bit = 8  # 16
normalization_factor = 2**use_bit

# Load splits and preprocess target
train_df, val_df, test_df, target_columns, preprocessor = get_dfs(args)

test_transform = transforms.Compose({
    ImagePreprocessor(normalization_factor=normalization_factor),
})
dataloader_test = setup_dataset(test_df, target_columns, test_transform,
                                args.batch_size, args.num_workers,
                                drop_last=False, shuffle=False)


preds, loss = test(dataloader_test, model, criterion, args.mode)

print(preds.shape)
preprocessor.inverse_transform(preds)

# torch.save(model.state_dict(), f'{args.save_dir}/model_%d.pth' % ep)
