deepspeed pre-train.py \
    --model_type="qwen" \
    --model_path="/mnt/file2/changye/model/NLP/Qwen2.5-1.5B-Instruct" \
    --train_data_path="/mnt/file2/changye/dataset/NLP/ACL_clear/train" \
    --val_data_path="/mnt/file2/changye/dataset/NLP/ACL_clear/val" \
    --output_path="/mnt/file2/changye/model/qwen-formal-trained" \
    --batch_size=4 \
    --epochs=3 \
    --lr=1e-5 \
    --max_length=512 \
    --save_every=500 \
    --deepspeed_config ds_config.json
