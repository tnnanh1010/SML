CHECKPOINT=${CHECKPOINT:-/root/code/train/output/checkpoint-last}

swift export \
    --adapters "$CHECKPOINT" \
    --merge_lora true
