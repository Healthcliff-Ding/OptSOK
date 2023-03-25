"""
 Copyright (c) 2021, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import tensorflow as tf
import numpy as np
from models import SOKDenseDemo
import argparse
import sys

sys.path.append("../")
import utility
import sparse_operation_kit as sok
import nvtx
import time

def main(args):
    strategy = tf.distribute.MirroredStrategy()

    dataset = utility.TFDataset(
        filename=args.data_filename,
        batchsize=args.global_batch_size,
        as_sparse_tensor=False,
        repeat=1,
    )
    # print(list(dataset.take(1)))
    # exit()
    data_option = tf.data.Options()
    data_option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    dataset = dataset.with_options(data_option)
    dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        sok.Init(global_batch_size=args.global_batch_size)

        model = SOKDenseDemo(
            max_vocabulary_size_per_gpu=args.max_vocabulary_size_per_gpu,
            embedding_vec_size=args.embedding_vec_size,
            slot_num=args.slot_num,
            nnz_per_slot=args.nnz_per_slot,
            num_dense_layers=args.num_dense_layers,
        )

        embedding_optimizer = utility.get_embedding_optimizer(args.optimizer)(learning_rate=0.1)
        dense_optimizer = utility.get_dense_optimizer(args.optimizer)(learning_rate=0.1)

    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    def _replica_loss(labels, logits):
        loss = loss_fn(labels, logits)
        return tf.nn.compute_average_loss(loss, global_batch_size=args.global_batch_size)

    @tf.function
    def _train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logit = model(inputs, training=True)
            loss = _replica_loss(labels, logit)
        emb_variable, other_variable = sok.split_embedding_variable_from_others(
            model.trainable_variables
        )
        grads, emb_grads = tape.gradient(loss, [other_variable, emb_variable])
        if "plugin" not in args.optimizer:
            with sok.OptimizerScope(emb_variable):
                embedding_optimizer.apply_gradients(
                    zip(emb_grads, emb_variable), experimental_aggregate_gradients=False
                )
        else:
            embedding_optimizer.apply_gradients(
                zip(emb_grads, emb_variable), experimental_aggregate_gradients=False
            )
        dense_optimizer.apply_gradients(zip(grads, other_variable))
        # return loss
        return loss, emb_grads[0].values[0][0], grads[-1]

    @tf.function
    def _warmup_step(inputs, labels):
        return inputs, labels

    warmup_list = []
    for inputs, labels in dataset:
        warm_data = strategy.run(_warmup_step, args=(inputs, labels))
        warmup_list.append(warm_data)

    # start = time.clock_gettime(time.CLOCK_MONOTONIC)
    # for inputs, labels in warmup_list:
    #     warm_data = strategy.run(_warmup_step, args=(inputs, labels))
    # end = time.clock_gettime(time.CLOCK_MONOTONIC)
    # print('[INFO]: iterate through warm-uped dataset {:.10f}'.format(end - start))

    time_list = []

    for e in range(args.epoch + 1):
        avg_time = 0.
        print('[INFO]: start epoch{}'.format(e))
        for i, (inputs, labels) in enumerate(warmup_list):
            if args.stop_at_iter > 0 and i >= args.stop_at_iter:
                break

            rng = nvtx.start_range(message="Iteration_" + str(i), color="blue")
            
            start = time.clock_gettime(time.CLOCK_MONOTONIC)
            # replica_loss = strategy.run(_train_step, args=(inputs, labels))
            replica_loss, _, _ = strategy.run(_train_step, args=(inputs, labels))
            # NOTE: make async execution end
            # NOTE: result of `local_result` is still PerReplica
            # grads_list = strategy.experimental_local_results(replica_grad)
            end = time.clock_gettime(time.CLOCK_MONOTONIC)
            loss = strategy.reduce(tf.distribute.ReduceOp.SUM, replica_loss, axis=None)
            nvtx.end_range(rng)
            # print("[INFO]: Iteration: {}, loss={:.10f}, time={:.10f}".format(i, loss, end - start))
            avg_time += end - start
        print('[INFO]: end epoch{}, average time: {:.10f}'.format(e, avg_time))
        time_list.append(avg_time)
    time_list = time_list[1:]
    print('[INFO]: Average time: {:.10f}, sigma ratio: {:.1f}%'.format(np.mean(time_list), np.std(time_list) / np.mean(time_list) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run DNN model with SparseOperationKit")

    parser.add_argument(
        "--data_filename", type=str, help="the filename of training datas", required=True
    )
    parser.add_argument("--global_batch_size", type=int, required=True)
    parser.add_argument("--max_vocabulary_size_per_gpu", type=int, required=True)
    parser.add_argument("--slot_num", type=int, required=True, help="the number of feature fields")
    parser.add_argument(
        "--nnz_per_slot", type=int, required=True, help="the number of keys in each slot"
    )
    parser.add_argument(
        "--num_dense_layers",
        type=int,
        required=True,
        help="the number of fully connected layers in this DNN model",
    )
    parser.add_argument(
        "--embedding_vec_size", type=int, required=True, help="the dimension of embedding vectors"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="use what optimizer",
        required=False,
        default="plugin_adam",
        choices=["plugin_adam", "adam", "sgd"],
    )
    parser.add_argument(
        "--stop_at_iter",
        type=int,
        required=False,
        help="early stop the process if iteration reachs this setting.",
        default=-1,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        default=6,
        help="number of epoch to train"
    )

    args = parser.parse_args()

    main(args)
