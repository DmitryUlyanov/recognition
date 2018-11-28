experiment_dir="extensions/qd/data/experiments/11-24,20:24;config_name:inception_v4_stage4,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data"
# experiment_dir="extensions/qd/data/experiments/11-24,08:50;config_name:cbam_resnet_stage2,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data"
part=val
checkpoint=model_34
device=7

CUDA_VISIBLE_DEVICES=${device} python test.py \
    --experiment_dir ${experiment_dir} \
    --checkpoint ${checkpoint}.pth \
    --part ${part} \
    --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_eval" \
    --save_driver npz_per_item \
    --set_eval_mode


# CUDA_VISIBLE_DEVICES=6 python test.py \
#     --experiment_dir ${experiment_dir} \
#     --checkpoint ${checkpoint}.pth \
#     --part ${part} \
#     --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_1" \
#     --save_driver npz_per_item \
#     --no-set_eval_mode \
#     --random_seed 1


# CUDA_VISIBLE_DEVICES=6 python test.py \
#     --experiment_dir ${experiment_dir} \
#     --checkpoint ${checkpoint}.pth \
#     --part ${part} \
#     --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_2" \
#     --save_driver npz_per_item \
#     --no-set_eval_mode \
#     --random_seed 2

# CUDA_VISIBLE_DEVICES=6 python test.py \
#     --experiment_dir ${experiment_dir} \
#     --checkpoint ${checkpoint}.pth \
#     --part ${part} \
#     --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_3" \
#     --save_driver npz_per_item \
#     --no-set_eval_mode \
#     --random_seed 3

# CUDA_VISIBLE_DEVICES=6 python test.py \
#     --experiment_dir ${experiment_dir} \
#     --checkpoint ${checkpoint}.pth \
#     --part ${part} \
#     --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_4" \
#     --save_driver npz_per_item \
#     --no-set_eval_mode \
#     --random_seed 4


CUDA_VISIBLE_DEVICES=${device} python test.py \
    --experiment_dir ${experiment_dir} \
    --checkpoint ${checkpoint}.pth \
    --part ${part} \
    --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_1_2048" \
    --save_driver npz_per_item \
    --no-set_eval_mode \
    --random_seed 1 \
    --batch_size 2048 


CUDA_VISIBLE_DEVICES=${device} python test.py \
    --experiment_dir ${experiment_dir} \
    --checkpoint ${checkpoint}.pth \
    --part ${part} \
    --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_2_2048" \
    --save_driver npz_per_item \
    --no-set_eval_mode \
    --random_seed 2 \
    --batch_size 2048

CUDA_VISIBLE_DEVICES=${device} python test.py \
    --experiment_dir ${experiment_dir} \
    --checkpoint ${checkpoint}.pth \
    --part ${part} \
    --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_3_1024" \
    --save_driver npz_per_item \
    --no-set_eval_mode \
    --random_seed 3 \
    --batch_size 1024

CUDA_VISIBLE_DEVICES=${device} python test.py \
    --experiment_dir ${experiment_dir} \
    --checkpoint ${checkpoint}.pth \
    --part ${part} \
    --dump_path "${experiment_dir}/${part}_preds_${checkpoint}_train_4_1024" \
    --save_driver npz_per_item \
    --no-set_eval_mode \
    --random_seed 4 \
    --batch_size 1024