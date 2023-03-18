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

#include "common.h"
#include "common/include/forward_functions.h"
#include "operation/operation_interface.h"

namespace SparseOperationKit {

template <typename EmbeddingType>
__global__ void reorderKernel(const size_t EmbeddingDimension, EmbeddingType const *inputs,
                              uint32_t const *indices, EmbeddingType *outputs, size_t,
                              size_t max_chunk_size, uint32_t const *chunk_sizes) {
  // set indices
  uint32_t gpu_idx = blockIdx.y;
  uint32_t thread_cnt = blockDim.x * blockDim.y;
  uint32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
  uint32_t curr_chunk_size = chunk_sizes[gpu_idx];
  // set shared memory
  extern __shared__ uint32_t idx_smem[];
  // set pointers and offsets
  uint32_t const *curr_input_idx = indices + gpu_idx * max_chunk_size;
  EmbeddingType const *curr_input_emb = inputs + gpu_idx * max_chunk_size * EmbeddingDimension;
  uint32_t size_per_block =
      (curr_chunk_size + gridDim.x * warpSize - 1) / (gridDim.x * warpSize) * warpSize;
  uint32_t lbound = blockIdx.x * size_per_block;
  uint32_t rbound = lbound + size_per_block;
  if (rbound > curr_chunk_size) {
    rbound = curr_chunk_size;
  }
  for (uint32_t offset = lbound; offset < rbound; offset += thread_cnt) {
    uint32_t curr_len = thread_cnt;
    if (offset + curr_len > rbound) {
      curr_len = rbound - offset;
    }
    if (thread_idx < curr_len) {
      idx_smem[thread_idx] = curr_input_idx[offset + thread_idx];
    }
    __syncthreads();
    for (uint32_t warp_idx = threadIdx.y; warp_idx < curr_len; warp_idx += blockDim.y) {
      uint32_t orig_idx = idx_smem[warp_idx];
      uint32_t pos_idx = offset + warp_idx;
      for (uint32_t elem_idx = threadIdx.x; elem_idx < EmbeddingDimension; elem_idx += blockDim.x) {
        outputs[orig_idx * EmbeddingDimension + elem_idx] =
            curr_input_emb[pos_idx * EmbeddingDimension + elem_idx];
      }
    }
    __syncthreads();
  }
}

template <typename EmbeddingType>
__global__ void gatherExKernel(const size_t EmbeddingDimension, EmbeddingType const *inputs,
                               uint32_t const *indices, EmbeddingType *outputs, size_t chunks,
                               size_t max_chunk_size, uint32_t const *chunk_sizes) {
  extern __shared__ uint32_t idx_smem[];
  uint32_t gpu_idx = blockIdx.y;
  uint32_t thread_cnt = blockDim.x * blockDim.y;
  uint32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
  uint32_t curr_chunk_size = chunk_sizes[gpu_idx];
  uint32_t const *curr_input_idx = indices + gpu_idx * max_chunk_size;
  EmbeddingType *curr_output = outputs + gpu_idx * max_chunk_size * EmbeddingDimension;
  uint32_t size_per_block =
      (curr_chunk_size + gridDim.x * warpSize - 1) / (gridDim.x * warpSize) * warpSize;
  uint32_t lbound = blockIdx.x * size_per_block;
  uint32_t rbound = lbound + size_per_block;
  if (rbound > curr_chunk_size) {
    rbound = curr_chunk_size;
  }
  for (uint32_t offset = lbound; offset < rbound; offset += thread_cnt) {
    uint32_t curr_len = thread_cnt;
    if (offset + curr_len > rbound) {
      curr_len = rbound - offset;
    }
    if (thread_idx < curr_len) {
      idx_smem[thread_idx] = curr_input_idx[offset + thread_idx];
    }
    __syncthreads();
    for (uint32_t warp_idx = threadIdx.y; warp_idx < curr_len; warp_idx += blockDim.y) {
      uint32_t pos_idx = offset + warp_idx;
      uint32_t orig_idx = idx_smem[warp_idx];
      for (uint32_t elem_idx = threadIdx.x; elem_idx < EmbeddingDimension; elem_idx += blockDim.x) {
        curr_output[pos_idx * EmbeddingDimension + elem_idx] =
            inputs[orig_idx * EmbeddingDimension + elem_idx];
      }
    }
    __syncthreads();
  }
}

template <typename EmbeddingType>
__global__ static void scatterGradKernel(const size_t EmbeddingDimension, EmbeddingType const *top_grad,
                                  uint32_t const *top_indices, EmbeddingType **replica_grad, 
                                  size_t chunks, size_t max_chunk_size, 
                                  uint32_t const *top_select_size, uint32_t const *replica_recv_offset) {
  uint32_t gpu_idx = blockIdx.y;
  uint32_t curr_chunk_size = top_select_size[gpu_idx];
  uint32_t const *curr_input_idx = top_indices + gpu_idx * max_chunk_size;
  EmbeddingType *curr_output = replica_grad[gpu_idx] + replica_recv_offset[gpu_idx] * EmbeddingDimension;

  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < curr_chunk_size * EmbeddingDimension;
       id += blockDim.x * gridDim.x) {
    size_t item_id = id / EmbeddingDimension;
    size_t embedding_id = id - item_id * EmbeddingDimension;

    size_t index = curr_input_idx[item_id];
    curr_output[id] = top_grad[index * EmbeddingDimension + embedding_id];
  }
}

template <typename ValueType>
class All2AllOutputDispatcher : public Dispatcher {
 public:
  explicit All2AllOutputDispatcher(ConstructionContext_t context)
      : Dispatcher(context),
        resource_mgr_(base_context()->get_resource_mgr()),
        num_keys_per_rank_(base_context()->get_replica_batch_size() *
                           base_context()->get_slot_num() * base_context()->get_nnz_per_slot()) {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    h_replica_input_grad_ptr_.reserve(local_gpu_count);
    h_replica_recv_chunk_offset_.reserve(local_gpu_count);
  }

  void allocate_forward_spaces() override {}


  void allocate_backward_spaces() override {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
      auto &buffer = base_context()->get_buffer(dev_id);
      auto &host_buffer = base_context()->get_host_buffer(dev_id);
      {
        Tensor2<ValueType*> tensor;
        host_buffer->reserve({global_gpu_count}, &tensor);
        h_replica_input_grad_ptr_.push_back(tensor);
      }
      {
        Tensor2<uint32_t> tensor;
        host_buffer->reserve({global_gpu_count}, &tensor);
        h_replica_recv_chunk_offset_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }

  void forward(const Context_t &replica_context, const bool training) override {}

  void backward(const Context_t &replica_context) override {
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    const auto &replica_top_gradients = replica_context->input("replica_top_gradient");
    const auto &replica_selected_indices_buf =
        replica_context->input("replica_selected_indices_buf");
    const auto &replica_num_selected_keys = replica_context->input("replica_num_selected_keys");
    const auto &replica_h_recv_chunk_offsets =
        replica_context->input("replica_h_recv_chunk_offsets");
    const auto &h_num_selected_keys = replica_context->input("replica_h_num_selected_keys");
    const auto &h_num_exchanged_keys = replica_context->input("replica_h_num_exchanged_keys");

    auto &replica_input_grad = replica_context->output("replica_input_grad");

    const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();

    //* step 1: issue gradient directly to where it ought to be
    //! assume local_gpu_cnt == global_gpu_cnt 
    //  use CPU thread to gather each peer's ptr
    for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
      h_replica_input_grad_ptr_[dev_id].get_ptr()[local_replica_id] = 
        replica_input_grad->GetPtrWithType<ValueType>();
      h_replica_recv_chunk_offset_[dev_id].get_ptr()[local_replica_id] =
        replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[dev_id];
    }
    resource_mgr_->sync_cpu_threads();
    //* WRITE style synchronize
    {
      dim3 const grid_dim(2 * local_gpu->get_sm_count() / global_gpu_count, global_gpu_count);
      scatterGradKernel<ValueType><<<grid_dim, 1024ul, 0, local_gpu->get_stream()>>>(
        embedding_vec_size, 
        replica_top_gradients->GetPtrWithType<ValueType>(),
        replica_selected_indices_buf->GetPtrWithType<uint32_t>(),
        h_replica_input_grad_ptr_[local_replica_id].get_ptr(),
        global_gpu_count,
        num_keys_per_rank_,
        replica_num_selected_keys->GetPtrWithType<uint32_t>(),
        h_replica_recv_chunk_offset_[local_replica_id].get_ptr());
      CK_CUDA(cudaGetLastError());  
    }
    CK_CUDA(cudaStreamSynchronize(local_gpu->get_stream()));
    resource_mgr_->sync_cpu_threads();
  }

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  const size_t num_keys_per_rank_;

  // forward spaces

  // backward spaces
  Tensors2<ValueType*> h_replica_input_grad_ptr_;
  Tensors2<uint32_t> h_replica_recv_chunk_offset_;
};

REGISTER_OUTPUT_DISPATHER_BUILDER("All2AllOutput", DataType::Int64, DataType::Float32,
                                  All2AllOutputDispatcher<float>);
REGISTER_OUTPUT_DISPATHER_BUILDER("All2AllOutput", DataType::Int64, DataType::Float16,
                                  All2AllOutputDispatcher<__half>);
REGISTER_OUTPUT_DISPATHER_BUILDER("All2AllOutput", DataType::Uint32, DataType::Float32,
                                  All2AllOutputDispatcher<float>);
REGISTER_OUTPUT_DISPATHER_BUILDER("All2AllOutput", DataType::Uint32, DataType::Float16,
                                  All2AllOutputDispatcher<__half>);

}  // namespace SparseOperationKit