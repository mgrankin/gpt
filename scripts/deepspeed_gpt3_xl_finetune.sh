#! /bin/bash

NUM_GPUS_PER_WORKER=1

gpt_options=" \
       --train-data-path dataset/pelevin_train.list \
       --val-data-path dataset/pelevin_valid.list \
       --eval-interval 100 \
       --max-files-per-process 20000 \
       --logging-dir=logs \
       --load-huggingface sberbank-ai/rugpt3xl \
       --save pelevin \
       --tokenizer-path sberbank-ai/rugpt3xl \
       --cache-prefix p5 \
       --save-interval 2300 \
       --no-load-optim \
       --finetune \
       --log-interval 100 \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --batch-size 2 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 2300 \
       --distributed-backend nccl \
       --lr 0.000003 \
       --warmup 0.0 \
       --lr-decay-style constant \
       --weight-decay 0.0 \
       --fp16 \
       --sparse-mode alternating \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --deepspeed \
       --deepspeed_config ../src/deepspeed_config/gpt3_xl_sparse_2048.json \
"

run_cmd="USE_DEEPSPEED=1 mpirun --np ${NUM_GPUS_PER_WORKER} python ../pretrain_gpt3.py $@ ${gpt_options}"
#run_cmd="USE_DEEPSPEED=1 python ../pretrain_gpt3.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
