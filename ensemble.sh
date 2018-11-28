# CBAM_RESNET
# python scripts/preds_to_submission.py \
#     --encoding ~/raid/dulyanov/qd/raw_data/classes_old.json \
#     --save_path ~/raid/dulyanov/qd/cbam_resnet__.csv \
#     --sample_submission ~/raid/dulyanov/qd/raw_data/sample_submission.csv \
#     --weights '0.3,0.3,0.1,0.1,0.1,0.1' \
#     --preds_dir "extensions/qd/data/experiments/11-24,08:50;config_name:cbam_resnet_stage2,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data/val_preds_model_69_eval" \
#     --preds_dir "extensions/qd/data/experiments/11-24,20:24;config_name:inception_v4_stage4,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data/val_preds_model_34_eval" \
    


python scripts/preds_to_submission.py \
    --encoding ~/raid/dulyanov/qd/raw_data/classes_old.json \
    --save_path ~/raid/dulyanov/qd/cbam_resnet_inception_v4_smail.csv \
    --sample_submission ~/raid/dulyanov/qd/raw_data/sample_submission.csv \
    --weights '0.5,0.5,0.5' \
    --preds_dir "../qd/data/preds_smail_test/" \
    --preds_dir "extensions/qd/data/experiments/11-24,08:50;config_name:cbam_resnet_stage2,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data/test_preds_model_69_eval" \
    # --preds_dir "extensions/qd/data/experiments/11-24,20:24;config_name:inception_v4_stage4,data_path:+raid+data+dulyanov+dulyanov+qd+raw_data/test_preds_model_34_eval" \

