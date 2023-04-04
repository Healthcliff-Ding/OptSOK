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

#include <sys/time.h>
#include "common/include/forward_functions.h"
#include "operation/operation_interface.h"

namespace SparseOperationKit {

static void profile_time(size_t id, int step, float cpu_time, float gpu_time) {
  std::cout << "step " << step << ": CPU " 
            << std::fixed << cpu_time / 100 << std::setprecision(9)
            << "ms GPU "  << gpu_time / 100 << "ms" << std::endl; 
}

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

template <typename ValueType>
class All2AllOutputDispatcher : public Dispatcher {
 public:
  explicit All2AllOutputDispatcher(ConstructionContext_t context)
      : Dispatcher(context),
        resource_mgr_(base_context()->get_resource_mgr()),
        num_keys_per_rank_(base_context()->get_replica_batch_size() *
                           base_context()->get_slot_num() * base_context()->get_nnz_per_slot()) {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    exchanged_embeddings_buf_.reserve(local_gpu_count);
    gathered_gradients_buf_.reserve(local_gpu_count);

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
      {
        Tensor2<ValueType> tensor;
        buffer->reserve({global_gpu_count, embedding_vec_size * num_keys_per_rank_}, &tensor);
        exchanged_embeddings_buf_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }

  void allocate_backward_spaces() override {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
      auto &buffer = base_context()->get_buffer(dev_id);

      {
        Tensor2<ValueType> tensor;  // FIXME: check whether top-grad is fp32 or fp16
        buffer->reserve({global_gpu_count, embedding_vec_size * num_keys_per_rank_}, &tensor);
        gathered_gradients_buf_.push_back(tensor);
      }
    }  // for dev_id in local_gpu_count
  }

  void forward(const Context_t &replica_context, const bool training) override {
    const size_t global_gpu_count = resource_mgr_->get_global_gpu_count();
    const size_t global_replica_id = replica_context->get_global_replica_id();
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    const auto &replica_gathered_embeddings = replica_context->input("replica_gathered_embeddings");
    const auto &h_recv_chunk_offsets = replica_context->input("replica_h_recv_chunk_offsets");
    const auto &h_num_exchanged_keys = replica_context->input("replica_h_num_exchanged_keys");
    const auto &h_num_selected_keys = replica_context->input("replica_h_num_selected_keys");
    const auto &replica_num_selected_keys = replica_context->input("replica_num_selected_keys");
    const auto &replica_selected_indices_buf =
        replica_context->input("replica_selected_indices_buf");

    auto &replica_output = replica_context->output("replica_output");

    // Profile Session
    timeval begin, end;
    cudaEventCreate(&start[local_replica_id]);
    cudaEventCreate(&stop[local_replica_id]);
    float cpu_time, gpu_time;
    cnt[local_replica_id][0]++;
    // step 1: exchange embedding values among all GPUs.
    // start profile
    gettimeofday(&begin, 0);
    cudaEventRecord(start[local_replica_id], local_gpu->get_stream());
    const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
    CK_NCCL(ncclGroupStart());
    for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
      CK_NCCL(ncclSend(
          replica_gathered_embeddings->GetPtrWithType<ValueType>() +
              h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
          h_num_exchanged_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
          GetNCCLType<ValueType>(), /*peer=*/dev_id, local_gpu->get_nccl(),
          local_gpu->get_stream()));
      CK_NCCL(ncclRecv(exchanged_embeddings_buf_[local_replica_id].get_ptr() +
                           dev_id * num_keys_per_rank_ * embedding_vec_size,
                       h_num_selected_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                       GetNCCLType<ValueType>(), /*peer=*/dev_id, local_gpu->get_nccl(),
                       local_gpu->get_stream()));
    }  // for dev_id in global_gpu_count
    CK_NCCL(ncclGroupEnd());
    cudaEventRecord(stop[local_replica_id], local_gpu->get_stream());
    cudaEventSynchronize(stop[local_replica_id]);
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][0] += cpu_time;
    gpu_time_acc[local_replica_id][0] += gpu_time;
    // end profile

    // step 2: reorder embedding values
    gettimeofday(&begin, 0);
    {
      // CK_CUDA(cudaMemsetAsync(replica_output->GetPtrWithType<float>(), 0,
      //                         replica_output->get_size_in_bytes(),
      //                         local_gpu->get_stream()));  // TODO: merge it to reorderKernel
      const size_t smem_size = local_gpu->get_max_smem_size_per_sm() / 2;
      CK_CUDA(cudaFuncSetAttribute(reorderKernel<ValueType>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      dim3 const grid_dim(2 * local_gpu->get_sm_count() / global_gpu_count, global_gpu_count);
      dim3 const block_dim(local_gpu->get_warp_size(), EMB_WARPS_PER_BLOCK);
      cudaEventRecord(start[local_replica_id], local_gpu->get_stream());
      reorderKernel<ValueType><<<grid_dim, block_dim, smem_size, local_gpu->get_stream()>>>(
          /*EmbeddingDimension=*/embedding_vec_size,
          /*inputs=*/exchanged_embeddings_buf_[local_replica_id].get_ptr(),
          /*indices=*/replica_selected_indices_buf->GetPtrWithType<uint32_t>(),
          /*outputs=*/replica_output->GetPtrWithType<ValueType>(),
          /*chunks=*/global_gpu_count,
          /*max_chunk_size=*/num_keys_per_rank_,
          /*chunk_sizes=*/replica_num_selected_keys->GetPtrWithType<uint32_t>());
      cudaEventRecord(stop[local_replica_id], local_gpu->get_stream());
      cudaEventSynchronize(stop[local_replica_id]);
      CK_CUDA(cudaGetLastError());
    }
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][1] += cpu_time;
    gpu_time_acc[local_replica_id][1] += gpu_time;
    // end profile
    // Profile Result
    if (cnt[local_replica_id][0] == 100) {
            auto session_name = []() {
        std::cout << "forward: Output Dispatcher" << std::endl;
      };
      resource_mgr_->blocking_call_once(session_name);
      for (int i = 0; i < 2; ++i) {
        resource_mgr_->one_at_a_time(profile_time, local_replica_id, 1 + i, 
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

    const auto &replica_top_gradients = replica_context->input("replica_top_gradient");
    const auto &replica_selected_indices_buf =
        replica_context->input("replica_selected_indices_buf");
    const auto &replica_num_selected_keys = replica_context->input("replica_num_selected_keys");
    const auto &replica_h_recv_chunk_offsets =
        replica_context->input("replica_h_recv_chunk_offsets");
    const auto &h_num_selected_keys = replica_context->input("replica_h_num_selected_keys");
    const auto &h_num_exchanged_keys = replica_context->input("replica_h_num_exchanged_keys");

    auto &replica_input_grad = replica_context->output("replica_input_grad");

    // Profile Session
    timeval begin, end;
    cudaEventCreate(&start[local_replica_id]);
    cudaEventCreate(&stop[local_replica_id]);
    float cpu_time, gpu_time;
    cnt[local_replica_id][1]++;

    // step 1: gather top gradients for local GPU.
    gettimeofday(&begin, 0);
    const size_t embedding_vec_size = base_context()->get_param()->get_embedding_vec_size();
    {
      const size_t smem_size = local_gpu->get_max_smem_size_per_sm() / 2;
      CK_CUDA(cudaFuncSetAttribute(gatherExKernel<ValueType>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      dim3 const grid_dim(2 * local_gpu->get_sm_count() / global_gpu_count, global_gpu_count);
      dim3 const block_dim(local_gpu->get_warp_size(), EMB_WARPS_PER_BLOCK);
      cudaEventRecord(start[local_replica_id], local_gpu->get_stream());
      gatherExKernel<ValueType><<<grid_dim, block_dim, smem_size, local_gpu->get_stream()>>>(
          /*EmbeddingDimension=*/embedding_vec_size,
          /*inputs=*/replica_top_gradients->GetPtrWithType<ValueType>(),
          /*indices=*/replica_selected_indices_buf->GetPtrWithType<uint32_t>(),
          /*outputs=*/gathered_gradients_buf_[local_replica_id].get_ptr(),
          /*chunks=*/global_gpu_count,
          /*max_chunk_size=*/num_keys_per_rank_,
          /*chunk_sizes=*/replica_num_selected_keys->GetPtrWithType<uint32_t>());
      cudaEventRecord(stop[local_replica_id], local_gpu->get_stream());
      cudaEventSynchronize(stop[local_replica_id]);
      CK_CUDA(cudaGetLastError());
    }
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][2] += cpu_time;
    gpu_time_acc[local_replica_id][2] += gpu_time;

    // step 2: exchange gradients among all GPUs.
    gettimeofday(&end, 0);
    cudaEventRecord(start[local_replica_id], local_gpu->get_stream());
    CK_NCCL(ncclGroupStart());
    for (size_t dev_id = 0; dev_id < global_gpu_count; dev_id++) {
      CK_NCCL(ncclSend(gathered_gradients_buf_[local_replica_id].get_ptr() +
                           dev_id * num_keys_per_rank_ * embedding_vec_size,
                       h_num_selected_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
                       GetNCCLType<ValueType>(), /*peer=*/dev_id, local_gpu->get_nccl(),
                       local_gpu->get_stream()));
      CK_NCCL(ncclRecv(
          replica_input_grad->GetPtrWithType<ValueType>() +
              replica_h_recv_chunk_offsets->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
          h_num_exchanged_keys->GetPtrWithType<uint32_t>()[dev_id] * embedding_vec_size,
          GetNCCLType<ValueType>(), /*peer=*/dev_id, local_gpu->get_nccl(),
          local_gpu->get_stream()));
    }  // for dev_id in global_gpu_count
    CK_NCCL(ncclGroupEnd());
    cudaEventRecord(stop[local_replica_id], local_gpu->get_stream());
    cudaEventSynchronize(stop[local_replica_id]);
    gettimeofday(&end, 0);
    cpu_time = (1000000.0 * (end.tv_sec - begin.tv_sec) + 
                end.tv_usec - begin.tv_usec)/1000.0;
    cudaEventElapsedTime(&gpu_time, start[local_replica_id], stop[local_replica_id]);
    cpu_time_acc[local_replica_id][3] += cpu_time;
    gpu_time_acc[local_replica_id][3] += gpu_time;
    
    // Profile Result
    if (cnt[local_replica_id][1] == 100) {
            auto session_name = []() {
        std::cout << "backward: Output Dispatcher" << std::endl;
      };
      resource_mgr_->blocking_call_once(session_name);
      for (int i = 0; i < 2; ++i) {
        resource_mgr_->one_at_a_time(profile_time, local_replica_id, 1 + i, 
                                                          cpu_time_acc[local_replica_id][i + 2], 
                                                          gpu_time_acc[local_replica_id][i + 2]);       
        cpu_time_acc[local_replica_id][i + 2] = 0.;
        gpu_time_acc[local_replica_id][i + 2] = 0.;                                                         
      }
      cnt[local_replica_id][1] = 0;
    }
  }

 private:
  std::shared_ptr<ResourcesManager> resource_mgr_;
  const size_t num_keys_per_rank_;

  // forward spaces
  Tensors2<ValueType> exchanged_embeddings_buf_;

  // backward spaces
  Tensors2<ValueType> gathered_gradients_buf_;

  // profile spaces
  size_t cnt[4][2];
  cudaEvent_t start[4], stop[4];
  float cpu_time_acc[4][4], gpu_time_acc[4][4];
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