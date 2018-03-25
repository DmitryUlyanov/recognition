Each file in `models` dir should implement `get_args` and `get_net` functions.


# Classificateion

Training for superheroes

```
CUDA_VISIBLE_DEVICES=2 python train.py --extension superheroes --image_size 480 --save_frequency 5 --patience 20 --num_epochs 200 --optimizer_args lr=3e-3^momentum=0.9 --augment --random_seed 133 --batch_size 16
```
