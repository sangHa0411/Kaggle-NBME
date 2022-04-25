# running
python train.py \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 16 \
--num_train_epochs 3 \
--learning_rate 3e-5 \
--weight_decay 1e-3 \
--warmup_ratio 0.05 \
--logging_strategy steps \
--save_strategy steps \
--evaluation_strategy steps \
--metric_for_best_model F1 \
--load_best_model_at_end F1