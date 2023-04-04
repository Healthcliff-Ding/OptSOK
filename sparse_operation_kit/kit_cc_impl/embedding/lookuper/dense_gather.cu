/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <bits/types/struct_timeval.h>
#include <sys/time.h>
#include "common.cuh"
#include "common/include/dumping_functions.h"
#include "common/include/forward_functions.h"
#include "hashtable/simple_hashtable.h"
#include "operation/operation_interface.h"

namespace SparseOperationKit {

static void profile_time(size_t id, int step, float cpu_time, float gpu_time) {
  std::cout << "step " << step << ": CPU " 
            << std::fixed << cpu_time / 100 << std::setprecision(9)
            << "ms GPU "  << gpu_time / 100 << "ms" << std::endl; 
}

template <typename EmbeddingType>
__global__ static void gatherKernel(const size_t EmbeddingDimension, float *__restrict__ inputs,
                                    size_t *indices, size_t num_indices, EmbeddingType *outputs) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_indices * EmbeddingDimension;
       id += blockDim.x * gridDim.x) {
    size_t item_id = id / EmbeddingDimension;
    size_t embedding_id = id - item_id * EmbeddingDimension;

    size_t index = static_cast<size_t>(indices[item_id]);
    outputs[id] = HugeCTR::TypeConvertFunc<EmbeddingType, float>::convert(
        inputs[index * EmbeddingDimension + embedding_id]);
  }
}

template <typename KeyType, typename ValueType>
class DenseGather : public EmbeddingLookuper {
 public:
  DenseGather(ConstructionContext_t context, std::shared_ptr<ParamInterface> param)
      : EmbeddingLookuper(context, param),
        resource_mgr_(base_context()->get_resource_mgr()),
        num_keys_per_rank_(base_context()->get_replica_batch_size() *
                           base_context()->get_slot_num() * base_context()->get_nnz_per_slot()) {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    mapped_indices_buf_.reserve(local_gpu_count);
    gathered_embeddings_buf_.reserve(local_gpu_count);

    if (sizeof(size_t) != sizeof(int64_t))
      throw std::runtime_error(
          "In this platform, sizeof(size_t) != sizeof(int64_t). "
          "It will cause unexpected behavoir when copy datas from "
          "size_t pointer to int64_t pointer.");

    if (param->get_hashtable(0)->identical_mapping()) {
      // identical_mapping waste memory spaces, so that lookuper
      // will set its wanted hashtable for param
      const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
      auto stream = resource_mgr_->get_local_gpu(0)->get_stream();
      const size_t capacity = param->get_hashtable(0)->get_capacity(stream);
      HashFunctor_t hash_func = HashFunctors::Divisive<KeyType, size_t>::create(
          /*interval=*/global_gpu_count, /*capacity=*/capacity,
          /*global_replica_id=*/resource_mgr_->cal_global_id_from_local_id(0));
      auto hashtable = SimpleHashtable<KeyType, size_t>::create(capacity, hash_func);
      param->set_hashtable(hashtable);
    }  // if identical_mapping

    // Profile Initialize
    for (int i = 0; i < 4; ++i) {
      cnt[i][0] = 0; cnt[i][1] = 0;
      for (int j = 0; j < 4; ++j) {
        cpu_time_acc[i][j] = 0.;
        gpu_time_acc[i][j] = 0.;
      }
    }
  }

  void allocate_forward_spaces() override {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
      auto &buffer = base_context()->get_buffer(dev_id);
      auto &host_buffer = base_context()->get_host_buffer(dev_id);
      {
        Tensor2<size_t> tensor;
        buffer->reserve({global_gpu_count, num_keys_per_rank_}, &tensor);
        mapped_indices_buf_.push_back(tensor);
      }
      {
        Tensor2<ValueType> tensor;
        buffer->reserve({global_gpu_count, embedding_vec_size * num_keys_per_rank_}, &tensor);
        gathered_embeddings_buf_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }
  void allocate_backward_spaces() override {}
  void forward(const Context_t &replica_context, const bool training) override {
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    auto &hashtable = param_->get_hashtable(local_replica_id);

    const auto &replica_exchanged_keys = replica_context->input("replica_exchanged_keys");
    const auto &replica_h_recv_chunk_offsets =
        replica_context->input("replica_h_recv_chunk_offsets");
    const uint32_t h_local_nnz =
        replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count];

    // Profile Session
    // each thread call once
    timeval begin, end;
    cudaEventCreate(&start[local_replica_id]);
    cudaEventCreate(&stop[local_replica_id]);
    float cpu_time, gpu_time;
    cnt[local_replica_id][0]++;

    // step 1: get index using keys
    // start profile
    gettimeofday(&begin, 0);
    cudaEventRecord(start[local_replica_id], local_gpu->get_stream());
    if (training) {
      hashtable->get_insert(replica_exchanged_keys->GetPtrWithType<KeyType>(),
                            mapped_indices_buf_[local_replica_id].get_ptr(),
                            /*nnz=*/h_local_nnz, local_gpu->get_stream());
    } else {
      hashtable->get(replica_exchanged_keys->GetPtrWithType<KeyType>(),
                     mapped_indices_buf_[local_replica_id].get_ptr(),
                     /*nnz=*/h_local_nnz, local_gpu->get_stream());
    }
    cudaEventRecord(stop[local_replica_id], local_gpu->get_stream());
    cudaEventSynchronize(stop[local_replica_id]);
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][0] += cpu_time;
    gpu_time_acc[local_replica_id][0] += gpu_time;
    // end profile

    // step 2: gather embedding vectors from embedding table
    // start profile
    gettimeofday(&begin, 0);
    cudaEventRecord(start[local_replica_id], local_gpu->get_stream());
    const auto &embedding_table = param_->get_embedding_table_tensor(local_replica_id);
    gatherKernel<ValueType><<<local_gpu->get_sm_count() * 2, 1024ul, 0, local_gpu->get_stream()>>>(
        /*EmbeddingDimension=*/param_->get_embedding_vec_size(),
        /*inputs=*/embedding_table->GetPtrWithType<float>(),
        /*indices=*/mapped_indices_buf_[local_replica_id].get_ptr(),
        /*num_indices=*/h_local_nnz,
        /*outputs=*/gathered_embeddings_buf_[local_replica_id].get_ptr());
    cudaEventRecord(stop[local_replica_id], local_gpu->get_stream());
    cudaEventSynchronize(stop[local_replica_id]);
    CK_CUDA(cudaGetLastError());
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][1] += cpu_time;
    gpu_time_acc[local_replica_id][1] += gpu_time;
    // end profile

    // step 3: set the output of embedding lookuper
    // start profile
    gettimeofday(&begin, 0);
    replica_context->set_output("replica_gathered_embeddings",
                                gathered_embeddings_buf_[local_replica_id]);
    // write host_nnz in current iteration
    auto &host_nnz = replica_context->output("replica_host_nnz");
    host_nnz->GetPtrWithType<size_t>()[0] = static_cast<size_t>(h_local_nnz);
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][2] += cpu_time;

    // Profile Result
    if (cnt[local_replica_id][0] == 100) {
      auto session_name = []() {
        std::cout << "forward: Dense Gather" << std::endl;
      };
      resource_mgr_->blocking_call_once(session_name);
      for (int i = 0; i < 3; ++i) {
        resource_mgr_->one_at_a_time(profile_time, local_replica_id, i + 1, 
                                                          cpu_time_acc[local_replica_id][i], 
                                                          gpu_time_acc[local_replica_id][i]);
        cpu_time_acc[local_replica_id][i] = 0.;
        gpu_time_acc[local_replica_id][i] = 0.;
      }
      cnt[local_replica_id][0] = 0;
    }
  }

  void backward(const Context_t &replica_context) override {
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    const auto &replica_h_recv_chunk_offsets =
        replica_context->input("replica_h_recv_chunk_offsets");
    const uint32_t h_local_nnz =
        replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count];
    auto &replica_value_index_tensor = replica_context->output("value_index_tensor");

    // FIXME: what if sizeof(size_t) != sizeof(int64_t)
    // Profile Session
    // each thread call once
    timeval begin, end;
    cudaEventCreate(&start[local_replica_id]);
    cudaEventCreate(&stop[local_replica_id]);
    float cpu_time, gpu_time;
    cnt[local_replica_id][1]++;
    // start profile
    gettimeofday(&begin, 0);
    cudaEventRecord(start[local_replica_id], local_gpu->get_stream());
    CK_CUDA(cudaMemcpyAsync(replica_value_index_tensor->GetPtrWithType<int64_t>(),
                            mapped_indices_buf_[local_replica_id].get_ptr(),
                            sizeof(size_t) * h_local_nnz, cudaMemcpyDeviceToDevice,
                            local_gpu->get_stream()));
    cudaEventRecord(stop[local_replica_id], local_gpu->get_stream());
    cudaEventSynchronize(stop[local_replica_id]);
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][3] += cpu_time;
    gpu_time_acc[local_replica_id][3] += gpu_time;
    // end profile
    // Profile Result
    if (cnt[local_replica_id][1] == 100) {
      auto session_name = []() {
        std::cout << "backward: Dense Gather" << std::endl;
      };
      resource_mgr_->blocking_call_once(session_name);
      resource_mgr_->one_at_a_time(profile_time, local_replica_id, 1, 
                                                        cpu_time_acc[local_replica_id][3], 
                                                        gpu_time_acc[local_replica_id][3]);
      cpu_time_acc[local_replica_id][3] = 0.;
      gpu_time_acc[local_replica_id][3] = 0.;
      
      cnt[local_replica_id][1] = 0;
    }
  }

  void save_params(std::shared_ptr<Tensor> &keys, std::shared_ptr<Tensor> &embedding_values,
                   size_t &num_total_keys) const override {
    // this lookuper distribute keys to each GPU based on key % GPU_NUM
    save_params_helper<KeyType>(param_, resource_mgr_, keys, embedding_values, num_total_keys);
  }

  void restore_params(const std::shared_ptr<Tensor> &keys,
                      const std::shared_ptr<Tensor> &embedding_values,
                      const size_t num_total_keys) override {
    // this lookuper distribute keys to each GPU based on key % GPU_NUM
    restore_params_helper<KeyType>(param_, resource_mgr_, keys, embedding_values, num_total_keys);
  }

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  const size_t num_keys_per_rank_;

  // forward spaces
  Tensors2<size_t> mapped_indices_buf_;
  Tensors2<ValueType> gathered_embeddings_buf_;

  // profile spaces
  size_t cnt[4][2];
  cudaEvent_t start[4], stop[4];
  float cpu_time_acc[4][4], gpu_time_acc[4][4];
};

REGISTER_EMB_LOOKUPER_BUILDER("dense_gather", DataType::Int64, DataType::Float32,
                              DenseGather<int64_t, float>);
REGISTER_EMB_LOOKUPER_BUILDER("dense_gather", DataType::Int64, DataType::Float16,
                              DenseGather<int64_t, __half>);
REGISTER_EMB_LOOKUPER_BUILDER("dense_gather", DataType::Uint32, DataType::Float32,
                              DenseGather<uint32_t, float>);
REGISTER_EMB_LOOKUPER_BUILDER("dense_gather", DataType::Uint32, DataType::Float16,
                              DenseGather<uint32_t, __half>);

}  // namespace SparseOperationKit