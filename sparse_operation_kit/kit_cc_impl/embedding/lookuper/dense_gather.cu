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

#include <cstddef>
#include <cstdint>
#include <iostream>
#include "common.cuh"
#include "common.h"
#include "common/include/dumping_functions.h"
#include "common/include/forward_functions.h"
#include "hashtable/simple_hashtable.h"
#include "operation/operation_interface.h"
#include "tensor_buffer/tensor_interface.h"

namespace SparseOperationKit {

template <typename KeyType, typename Hasher, typename EmbeddingType>
__global__ static void gatherKernel(const size_t EmbeddingDimension, float **__restrict__ emb_tbl_ptr,
                                    KeyType const *keys, size_t *mapped_indices, const size_t num_keys, 
                                    EmbeddingType *outputs, const size_t chunks) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_keys * EmbeddingDimension;
       id += blockDim.x * gridDim.x) {
    size_t item_id = id / EmbeddingDimension;
    size_t embedding_id = id - item_id * EmbeddingDimension;
    size_t chunk_id = Hasher::compute(keys[item_id]) % chunks;

    size_t index = static_cast<size_t>(mapped_indices[item_id]);
    outputs[id] = HugeCTR::TypeConvertFunc<EmbeddingType, float>::convert(
        emb_tbl_ptr[chunk_id][index * EmbeddingDimension + embedding_id]);
  }
}

template <typename KeyType>
__global__ static void scatterIdxKernel(KeyType const *chunk_keys, uint32_t const *chunk_indices,
                                  size_t const *chunk_mapped_offset, size_t **replica_mapped_offset,
                                  size_t chunks, size_t max_chunk_size, 
                                  uint32_t const *replica_select_size, uint32_t const *replica_recv_offset) {
  uint32_t gpu_idx = blockIdx.y;
  uint32_t curr_chunk_size = replica_select_size[gpu_idx];
  uint32_t const *curr_input_idx = chunk_indices + gpu_idx * max_chunk_size;
  size_t *curr_output = replica_mapped_offset[gpu_idx] + replica_recv_offset[gpu_idx];

  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < curr_chunk_size;
       id += blockDim.x * gridDim.x) {

    size_t index = curr_input_idx[id];
    curr_output[id] = chunk_mapped_offset[index];
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
    h_emb_tbl_ptr_.reserve(local_gpu_count);
    h_mapped_indices_ptr_.reserve(local_gpu_count);
    h_replica_recv_chunk_offset_.reserve(local_gpu_count);

    if (sizeof(size_t) != sizeof(int64_t))
      throw std::runtime_error(
          "In this platform, sizeof(size_t) != sizeof(int64_t). "
          "It will cause unexpected behavoir when copy datas from "
          "size_t pointer to int64_t pointer.");

    if (param->get_hashtable(0)->identical_mapping()) {
      // identical_mapping waste memory spaces, so that lookuper
      // will set its wanted hashtable for param
      MESSAGE("[INFO]: use Simple HashTable");
      const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
      auto stream = resource_mgr_->get_local_gpu(0)->get_stream();
      const size_t capacity = param->get_hashtable(0)->get_capacity(stream);
      HashFunctor_t hash_func = HashFunctors::Divisive<KeyType, size_t>::create(
          /*interval=*/global_gpu_count, /*capacity=*/capacity,
          /*global_replica_id=*/resource_mgr_->cal_global_id_from_local_id(0));
      auto hashtable = SimpleHashtable<KeyType, size_t>::create(capacity, hash_func);
      param->set_hashtable(hashtable);
    }  // if identical_mapping
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
        Tensor2<float*> tensor;
        host_buffer->reserve({global_gpu_count}, &tensor);
        h_emb_tbl_ptr_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }

  void allocate_backward_spaces() override {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
      auto &host_buffer = base_context()->get_host_buffer(dev_id);
      {
        Tensor2<size_t*> tensor;
        host_buffer->reserve({global_gpu_count}, &tensor);
        h_mapped_indices_ptr_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        host_buffer->reserve({global_gpu_count}, &tensor);
        h_replica_recv_chunk_offset_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }

  void forward(const Context_t &replica_context, const bool training) override {
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    auto &hashtable = param_->get_hashtable(local_replica_id);

    const auto &input_keys = replica_context->input("replica_values");
    const auto &replica_h_recv_chunk_offsets =
        replica_context->input("replica_h_recv_chunk_offsets");
    const uint32_t h_local_nnz =
        replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count];
    auto &replica_output = replica_context->output("replica_output");

    // step 1: get index using keys
    if (training) {
      hashtable->get_insert(input_keys->GetPtrWithType<KeyType>(),
                            mapped_indices_buf_[local_replica_id].get_ptr(),
                            /*nnz=*/input_keys->get_num_elements(), local_gpu->get_stream());
    } else {
      hashtable->get(input_keys->GetPtrWithType<KeyType>(),
                     mapped_indices_buf_[local_replica_id].get_ptr(),
                     /*nnz=*/input_keys->get_num_elements(), local_gpu->get_stream());
    }

    // step 2: gather embedding vectors from embedding table
    //! assume local_gpu_count == global_gpu_count
    for (size_t dev_id = 0; dev_id < global_gpu_count; ++dev_id) {
      const auto &embedding_table = param_->get_embedding_table_tensor(dev_id);
      h_emb_tbl_ptr_[local_replica_id].get_ptr()[dev_id] = embedding_table->GetPtrWithType<float>();
    }

    //* READ style synchronize
    //  once Embedding table is set, no need for CPU barrier
    gatherKernel<KeyType, IdenticalHash, ValueType><<<local_gpu->get_sm_count() * 2, 1024ul, 0, local_gpu->get_stream()>>>(
        param_->get_embedding_vec_size(),
        h_emb_tbl_ptr_[local_replica_id].get_ptr(),
        input_keys->GetPtrWithType<KeyType>(),
        mapped_indices_buf_[local_replica_id].get_ptr(),
        input_keys->get_num_elements(),
        replica_output->GetPtrWithType<ValueType>(),
        global_gpu_count
    );
    CK_CUDA(cudaGetLastError());

    // write host_nnz in current iteration
    auto &host_nnz = replica_context->output("replica_host_nnz");
    host_nnz->GetPtrWithType<size_t>()[0] = static_cast<size_t>(h_local_nnz);
  }

  void backward(const Context_t &replica_context) override {
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    const auto &replica_exchanged_key = replica_context->input("replica_exchanged_keys");
    const auto &replica_selected_indices = replica_context->input("replica_selected_indices_buf");
    const auto &replica_num_selected_keys = replica_context->input("replica_num_selected_keys");
    const auto &replica_h_recv_chunk_offsets =
        replica_context->input("replica_h_recv_chunk_offsets");
    const uint32_t h_local_nnz =
        replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[global_gpu_count];
    auto &replica_value_index_tensor = replica_context->output("value_index_tensor");

    //* Key Exchanged
    // {
    //   auto &hashtable = param_->get_hashtable(local_replica_id);
    //   hashtable->get_insert(replica_exchanged_key->GetPtrWithType<KeyType>(), 
    //                         replica_value_index_tensor->GetPtrWithType<size_t>(), 
    //                         h_local_nnz, local_gpu->get_stream());
    //   CK_CUDA(cudaMemcpyAsync(replica_value_index_tensor->GetPtrWithType<int64_t>(),
    //                           mapped_indices_buf_[local_replica_id].get_ptr(),
    //                           sizeof(size_t) * h_local_nnz, cudaMemcpyDeviceToDevice,
    //                           local_gpu->get_stream()));
    // }
    
    // //* P2P direct write 
    {
      //! assume local_gpu_cnt == global_gpu_cnt
      for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
        h_mapped_indices_ptr_[dev_id].get_ptr()[local_replica_id] = 
          replica_value_index_tensor->GetPtrWithType<size_t>();
        h_replica_recv_chunk_offset_[dev_id].get_ptr()[local_replica_id] =
          replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[dev_id];
      }
      resource_mgr_->sync_cpu_threads();
      //! assume sizeof(size_t) == sizeof(int64_t)
      //* WRITE style synchronize
      {
        dim3 const grid_dim(2 * local_gpu->get_sm_count() / global_gpu_count, global_gpu_count);
        scatterIdxKernel<KeyType><<<grid_dim, 1024ul, 0, local_gpu->get_stream()>>>(
          replica_exchanged_key->GetPtrWithType<KeyType>(), 
          replica_selected_indices->GetPtrWithType<uint32_t>(),
          mapped_indices_buf_[local_replica_id].get_ptr(),
          h_mapped_indices_ptr_[local_replica_id].get_ptr(),
          global_gpu_count,
          num_keys_per_rank_,
          replica_num_selected_keys->GetPtrWithType<uint32_t>(),
          h_replica_recv_chunk_offset_[local_replica_id].get_ptr());
        CK_CUDA(cudaGetLastError());
      }
      CK_CUDA(cudaStreamSynchronize(local_gpu->get_stream()));
      resource_mgr_->sync_cpu_threads();
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
  Tensors2<float*> h_emb_tbl_ptr_;

  // backward spaces
  Tensors2<size_t*> h_mapped_indices_ptr_;
  Tensors2<uint32_t> h_replica_recv_chunk_offset_;
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