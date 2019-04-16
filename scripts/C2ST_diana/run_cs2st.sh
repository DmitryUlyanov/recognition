CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/dcgan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/dfi --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/l50 --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/plain --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1

CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/dcgan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/dfi --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/l50 --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/smile/splits/plain --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2

CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/dcgan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/dfi   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/l50   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/plain --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1

CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/dcgan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/dfi   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/l50   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/plain --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2



CUDA_VISIBLE_DEVICES=0 python train.py --splits_dir data/C2ST_diana/smile/splits/cyclegan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1 &
CUDA_VISIBLE_DEVICES=1 python train.py --splits_dir data/C2ST_diana/smile/splits/cyclegan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2 &
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/m-f/splits/cyclegan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1 &
CUDA_VISIBLE_DEVICES=3 python train.py --splits_dir data/C2ST_diana/m-f/splits/cyclegan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2 &

CUDA_VISIBLE_DEVICES=0 python train.py --splits_dir data/C2ST_diana/brown/splits/dcgan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1 &
CUDA_VISIBLE_DEVICES=1 python train.py --splits_dir data/C2ST_diana/brown/splits/dfi   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1 &
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/brown/splits/l50   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1 &
CUDA_VISIBLE_DEVICES=3 python train.py --splits_dir data/C2ST_diana/brown/splits/plain --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1 &

CUDA_VISIBLE_DEVICES=0 python train.py --splits_dir data/C2ST_diana/brown/splits/cyclegan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run1 &
CUDA_VISIBLE_DEVICES=1 python train.py --splits_dir data/C2ST_diana/brown/splits/dcgan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2 &
CUDA_VISIBLE_DEVICES=2 python train.py --splits_dir data/C2ST_diana/brown/splits/dfi   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2 &
CUDA_VISIBLE_DEVICES=3 python train.py --splits_dir data/C2ST_diana/brown/splits/l50   --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2 &
CUDA_VISIBLE_DEVICES=0 python train.py --splits_dir data/C2ST_diana/brown/splits/plain --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2 &
CUDA_VISIBLE_DEVICES=1 python train.py --splits_dir data/C2ST_diana/brown/splits/cyclegan --arch resnet18 --mode classification --model resnet_classification_fixed --dataloader C2ST_diana --num_epochs 20 --comment run2 &


