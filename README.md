# A general pipeline 

1. Create a new extension for your project. 
```bash 
bash scripts/add_extension.sh my_extension
```
This creates a following folder structure: 

```
extensions/my_extension/
├── dataloaders
├── models
├── runners
├── scripts
└── config.yaml
```

2. Define your dataloaders, models, runners, scripts in corresponding directories.
    - Each file in `models` dir should implement `get_args`, `get_net`, `get_native_transform` functions.
    - ...

    Use `classification.py` dataloader/runner as example and `resnet_classification.py` as an exaple of a model.

Classification
--------------

Here is an easy way to solve a classification problem. 

1. Prepare your dataset to have this structure: 
```
dataset
├── class1
├── class2
└── class3
```
Images for each class in a corresponding folder.

2. Run this script to gather images and split into train and validation.
```bash
python scripts/train_test_split.py --classes_dir path/to/dataset --save_dir extensions/my_extension/data/splits --test_size 0.
```

3. Learn something
```
python train.py --config_name classification
```
