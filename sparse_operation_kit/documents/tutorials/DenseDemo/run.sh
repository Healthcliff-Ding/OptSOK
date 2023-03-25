#!/usr/bin/bash
# python3 gen_data.py \
#     --global_batch_size=4 \
#     --slot_num=1 \
#     --nnz_per_slot=1 \
#     --iter_num=1 \
#     --filename="min2.file" 
python3 run_sok_MirroredStrategy.py \
    --data_filename="./max.file" \
    --global_batch_size=32768 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=100 \
    --nnz_per_slot=1 \
    --num_dense_layers=3 \
    --embedding_vec_size=256 \
    --optimizer="plugin_adam" \
    --epoch=$1
# min2: bz:4, slot:1
# min1: bz:8, slot:2
# other: bz:65536, slot:100