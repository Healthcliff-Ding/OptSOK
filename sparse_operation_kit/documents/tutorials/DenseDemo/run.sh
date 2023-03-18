#!/usr/bin/bash
# python3 gen_data.py \
#     --global_batch_size=65536 \
#     --slot_num=100 \
#     --nnz_per_slot=1 \
#     --iter_num=50 
python3 run_sok_MirroredStrategy.py \
    --data_filename="./max.file" \
    --global_batch_size=65536 \
    --max_vocabulary_size_per_gpu=8192 \
    --slot_num=100 \
    --nnz_per_slot=1 \
    --num_dense_layers=3 \
    --embedding_vec_size=32 \
    --optimizer="adam" 