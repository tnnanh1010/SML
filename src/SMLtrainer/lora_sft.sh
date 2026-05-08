DATASET=${DATASET:-/root/code/dataset.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/root/code.SMLtrainer/output}

# --loss_scale: set to ignore_empty_thinking for training thinking model
# --attn_impl: set to flash_attention if downloaded
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model Qwen/Qwen3.5-4B \
    --use_hf true \
    --dataset "$DATASET" \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 512 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 6 \
    --attn_impl sdpa \
    --output_dir "$OUTPUT_DIR" \
    --save_strategy steps \
    --early_stop_interval 3 \
    --eval_steps 30 \
    --save_steps 30 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 8 \
    --load_from_cache_file true \
    --model_author swift \
    --model_name swift-robot \
    --loss_scale default \
    --report_to wandb