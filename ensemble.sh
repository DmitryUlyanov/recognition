# CBAM_RESNET
# python scripts/preds_to_submission.py \
#     --encoding /sdf/kaggle_data/quickdraw/raw_data/classes_old.json \
#     --save_path /sdf/kaggle_data/quickdraw/data/temp.csv \
#     --sample_submission /sdf/kaggle_data/quickdraw/raw_data/sample_submission.csv \
#     --weights '0.5,0.25,0.25' \
#     --preds_dir "extensions/quickdraw/data/experiments/11-25,10:30;config_name:inception_v4_smail_stage3,data_path:+sdf+kaggle_data+quickdraw+raw_data/val_preds_model_25_eval" \
    # --preds_dir "extensions/quickdraw/data/experiments/11-25,10:30;config_name:inception_v4_smail_stage3,data_path:+sdf+kaggle_data+quickdraw+raw_data/val_preds_model_25_train_1_2048" \
    # --preds_dir "extensions/quickdraw/data/experiments/11-25,10:30;config_name:inception_v4_smail_stage3,data_path:+sdf+kaggle_data+quickdraw+raw_data/val_preds_model_25_train_3_1024" \
    


python scripts/preds_to_submission.py \
    --encoding /sdf/kaggle_data/quickdraw/raw_data/classes_old.json \
    --save_path /sdf/kaggle_data/quickdraw/data/inception_smail.csv \
    --sample_submission /sdf/kaggle_data/quickdraw/raw_data/sample_submission.csv \
    --weights '1.0' \
    --preds_dir "extensions/quickdraw/data/experiments/11-25,10:30;config_name:inception_v4_smail_stage3,data_path:+sdf+kaggle_data+quickdraw+raw_data/test_preds_model_25_eval" \


# python scripts/preds_to_submission.py \
#     --encoding ~/raid/dulyanov/qd/raw_data/classes_old.json \
#     --save_path ~/raid/dulyanov/qd/cbam_resnet_inception_v4_8908.csv \
#     --sample_submission ~/raid/dulyanov/qd/raw_data/sample_submission.csv \
#     --weights '0.5,0.5' \
#     --preds_dir "extensions/qd/data/experiments/11-24,08:50;config_name:cbam_resnet_stage2,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data/test_preds_model_69_eval" \
#     --preds_dir "extensions/qd/data/experiments/11-24,20:24;config_name:inception_v4_stage4,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data/test_preds_model_34_eval" \
