CHECKPOINT=${CHECKPOINT:-/root/code.SMLtrainer/output/checkpoint-last}

swift export \
    --adapters "$CHECKPOINT" \
    --merge_lora true
