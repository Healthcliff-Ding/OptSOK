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

#include "HugeCTR/include/layers/multi_head_attention_layer.hpp"

#include <cublas_v2.h>
#include <gtest/gtest.h>
#include <math.h>
#include <utest/test_utils.h>

#include <memory>
#include <utils.hpp>
#include <vector>

#include "HugeCTR/include/utils.hpp"
using namespace std;
using namespace HugeCTR;

namespace {

template <typename T>
T get_eps(bool use_tf32 = false);

template <>
float get_eps(bool use_tf32) {
  return (use_tf32 ? 5e-1 : 1e-3);
}

template <>
__half get_eps(bool use_tf32) {
  return __float2half(1);
}

template <typename T>
void matmul_cpu(T *in1, T *in2, T *output, size_t B, size_t H, size_t M, size_t N, size_t K) {
  //(m,n)x(n,k)
  size_t i, j, k, z, y;
  for (y = 0; y < B; y++) {
    for (z = 0; z < H; z++) {
      for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++) {
          output[y * H * M * K + z * M * K + i * K + j] = TypeConvert<T, float>::convert(0.0f);
          for (k = 0; k < N; k++) {
            output[y * H * M * K + z * M * K + i * K + j] =
                output[y * H * M * K + z * M * K + i * K + j] +
                in1[y * H * M * N + z * M * N + i * N + k] *
                    in2[y * H * N * K + z * N * K + k * K + j];
          }
        }
      }
    }
  }
}

//[b, h, m, n] -> [b, h, n, m]
template <typename T>
static void transpose(T *a, size_t b, size_t h, size_t m, size_t n) {
  std::unique_ptr<T[]> tmp(new T[m * n]);
  for (size_t y = 0; y < b; y++) {
    for (size_t z = 0; z < h; z++) {
      T *cur_a = a + z * m * n + y * h * m * n;
      for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) tmp[j * m + i] = cur_a[i * n + j];
      for (size_t i = 0; i < m * n; ++i) cur_a[i] = tmp[i];
    }
  }
}

// transpose from [batch_size, seq_len, hidden_num] -> [batch_size, head_num, seq_len,
// size_per_head]
template <typename T>
void transpose_v_cpu(T *value, size_t batch_size, size_t seq_len, size_t head_num,
                     size_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_v(new T[size]);
  for (size_t b = 0; b < batch_size; b++) {
    T *cur_v = value + b * seq_len * hidden_dim;
    for (size_t s = 0; s < seq_len; s++) {
      for (size_t d = 0; d < hidden_dim; d++) {
        tmp_v[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
              d % size_per_head] = cur_v[s * hidden_dim + d];
      }
    }
    for (size_t i = 0; i < size; ++i) {
      cur_v[i] = tmp_v[i];
    }
  }
}

template <typename T>
void transpose_qkv_cpu(T *query, T *key, T *value, T *value_4d, size_t batch_size, size_t seq_len,
                       size_t head_num, size_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_q(new T[size]);
  std::unique_ptr<T[]> tmp_k(new T[size]);
  for (size_t b = 0; b < batch_size; b++) {
    T *cur_q = query + b * seq_len * hidden_dim;
    T *cur_k = key + b * seq_len * hidden_dim;
    T *cur_v = value + b * seq_len * hidden_dim;
    T *cur_v_4d = value_4d + b * seq_len * hidden_dim;
    for (size_t s = 0; s < seq_len; s++) {
      for (size_t d = 0; d < hidden_dim; d++) {
        tmp_q[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
              d % size_per_head] = cur_q[s * hidden_dim + d];
        tmp_k[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
              d % size_per_head] = cur_k[s * hidden_dim + d];
        cur_v_4d[(d / size_per_head) * seq_len * size_per_head + s * size_per_head +
                 d % size_per_head] = cur_v[s * hidden_dim + d];
      }
    }
    for (size_t i = 0; i < size; ++i) {
      cur_q[i] = tmp_q[i];
      cur_k[i] = tmp_k[i];
    }
  }
}

// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
template <typename T>
void transpose_qkv_back_cpu(T *query, T *key, T *value, T *value_4d, size_t batch_size,
                            size_t seq_len, size_t head_num, size_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_q(new T[size]);
  std::unique_ptr<T[]> tmp_k(new T[size]);
  for (size_t b = 0; b < batch_size; b++) {
    T *cur_q = query + b * seq_len * hidden_dim;
    T *cur_k = key + b * seq_len * hidden_dim;
    T *cur_v = value + b * seq_len * hidden_dim;
    T *cur_v_4d = value_4d + b * seq_len * hidden_dim;
    for (size_t h = 0; h < head_num; h++) {
      for (size_t s = 0; s < seq_len; s++) {
        for (size_t d = 0; d < size_per_head; d++) {
          tmp_q[s * hidden_dim + h * size_per_head + d] =
              cur_q[h * seq_len * size_per_head + s * size_per_head + d];
          tmp_k[s * hidden_dim + h * size_per_head + d] =
              cur_k[h * seq_len * size_per_head + s * size_per_head + d];
          cur_v[s * hidden_dim + h * size_per_head + d] =
              cur_v_4d[h * seq_len * size_per_head + s * size_per_head + d];
        }
      }
    }
    for (size_t i = 0; i < size; ++i) {
      cur_q[i] = tmp_q[i];
      cur_k[i] = tmp_k[i];
    }
  }
}

// transpose from [batch_size, head_num, seq_len, size_per_head] -> [batch_size, seq_len,
// hidden_num]
template <typename T>
void transpose_v_back_cpu(T *value, size_t batch_size, size_t seq_len, size_t head_num,
                          size_t hidden_dim) {
  auto size = seq_len * hidden_dim;
  auto size_per_head = hidden_dim / head_num;
  std::unique_ptr<T[]> tmp_v(new T[size]);
  for (size_t b = 0; b < batch_size; b++) {
    T *cur_v = value + b * seq_len * hidden_dim;
    for (size_t h = 0; h < head_num; h++) {
      for (size_t s = 0; s < seq_len; s++) {
        for (size_t d = 0; d < size_per_head; d++) {
          tmp_v[s * hidden_dim + h * size_per_head + d] =
              cur_v[h * seq_len * size_per_head + s * size_per_head + d];
        }
      }
    }
    for (size_t i = 0; i < size; ++i) {
      cur_v[i] = tmp_v[i];
    }
  }
}

template <typename T>
void multi_head_attention_cpu(T *in1, T *in2, T *output, size_t b, size_t h, size_t m, size_t n,
                              size_t k) {
  transpose(in2, b, h, n, k);
  matmul_cpu(in1, in2, output, b, h, m, k, n);
  // Just to revert in2 back
  transpose(in2, b, h, k, n);
  for (int i = 0; i < b * h * m * n; i++) {
    output[i] = output[i] / ((float)sqrt(k));
  }
}

template <typename T>
void multi_head_attention_cpu_noT(T *in1, T *in2, T *output, size_t b, size_t h, size_t m, size_t n,
                                  size_t k) {
  matmul_cpu(in1, in2, output, b, h, m, n, k);
  transpose_v_back_cpu(output, b, m, h, h * k);
}

template <typename T>
void multi_head_attention_3d_cpu(T *in1, T *in2, T *in3, T *output, T *value_out, size_t batch_size,
                                 size_t seq_len, size_t hidden_dim, size_t head_num) {
  transpose_qkv_cpu(in1, in2, in3, value_out, batch_size, seq_len, head_num, hidden_dim);
  multi_head_attention_cpu(in1, in2, output, batch_size, head_num, seq_len, seq_len,
                           hidden_dim / head_num);
}

template <typename T>
void multi_head_attention_dgrad_cpu(T *out, T **h_ins, T **h_b_ins, size_t b, size_t h, size_t m,
                                    size_t n, size_t k) {
  // transpose(h_ins[1], h, b, n, k);
  // transpose(h_ins[0], h, b, m, n);
  // out [b,h,m,n]
  // in1 [b,h,m,k]
  // in2 [b,h,n,k]
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, h, m, n, k);
  transpose(out, b, h, m, n);
  matmul_cpu(out, h_ins[0], h_b_ins[1], b, h, n, m, k);
  // Just revert out back
  transpose(out, b, h, m, n);
  for (int i = 0; i < b * h * m * k; i++) {
    h_b_ins[0][i] = h_b_ins[0][i] / ((float)sqrt(k));
  }
  for (int i = 0; i < b * h * n * k; i++) {
    h_b_ins[1][i] = h_b_ins[1][i] / ((float)sqrt(k));
  }
}
template <typename T>
void multi_head_attention_dgrad_3d_cpu(T *out, T *value_out, T **h_ins, T **h_b_ins,
                                       size_t batch_size, size_t head_num, size_t seq_len,
                                       size_t hidden_dim) {
  multi_head_attention_dgrad_cpu(out, h_ins, h_b_ins, batch_size, head_num, seq_len, seq_len,
                                 hidden_dim / head_num);
  transpose_qkv_back_cpu(h_b_ins[0], h_b_ins[1], h_b_ins[2], value_out, batch_size, seq_len,
                         head_num, hidden_dim);
}

template <typename T>
void multi_head_attention_dgrad_cpu_noT(T *out, T **h_ins, T **h_b_ins, size_t b, size_t h,
                                        size_t m, size_t n, size_t k) {
  transpose_v_cpu(out, b, m, h, h * k);
  transpose(h_ins[1], b, h, n, k);
  matmul_cpu(out, h_ins[1], h_b_ins[0], b, h, m, k, n);
  transpose(h_ins[0], b, h, m, n);
  matmul_cpu(h_ins[0], out, h_b_ins[1], b, h, n, m, k);
}
template <typename T>
void multi_head_attention_layer_test_4d(size_t batch_size, size_t head_num, size_t from_seq_len,
                                        size_t to_seq_len, size_t size_per_head,
                                        bool use_mixed_precision = false,
                                        bool enable_tf32_compute = false) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  size_t out_size = batch_size * head_num * from_seq_len * to_seq_len;
  int dims = 4;

  Tensors2<T> in_tensors;
  Tensor2<T> in_tensor;

  buff->reserve({batch_size, head_num, from_seq_len, size_per_head}, &in_tensor);
  in_tensors.push_back(in_tensor);
  buff->reserve({batch_size, head_num, to_seq_len, size_per_head}, &in_tensor);
  in_tensors.push_back(in_tensor);
  Tensors2<T> out_tensors;

  MultiHeadAttentionLayer multi_head_attention_layer(in_tensors, out_tensors, buff, head_num, true,
                                                     test::get_default_gpu(), use_mixed_precision,
                                                     enable_tf32_compute);

  buff->allocate();

  size_t num = 2;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  for (size_t i = 0; i < 2; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }

  // test 4d input for mha layer
  for (size_t i = 0; i < num; i++) {
    size_t size = batch_size * head_num * in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    h_ins[i] = new T[size];
    h_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    simulator.fill(h_ins[i], test::align_to_even(size));
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::shared_ptr<GPUResource> gpu_resource = test::get_default_gpu();
  cudaStreamSynchronize(gpu_resource->get_stream());
  std::cout << cudaGetLastError() << std::endl;

  T *d_out = out_tensors[0].get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(h_out.get(), d_out, out_tensors[0].get_size_in_bytes(), cudaMemcpyDeviceToHost));
  multi_head_attention_cpu(h_ins[0], h_ins[1], h_cpu_out.get(), batch_size, head_num, from_seq_len,
                           to_seq_len, size_per_head);

  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size,
                                            get_eps<T>(enable_tf32_compute)));

  // device bprop
  simulator.fill(h_out.get(), test::align_to_even(out_size));
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), out_size * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  multi_head_attention_dgrad_cpu(h_out.get(), h_ins.get(), h_cpu_bprop_out.get(), batch_size,
                                 head_num, from_seq_len, to_seq_len, size_per_head);
  for (size_t i = 0; i < num; i++) {
    size_t size = batch_size * head_num * in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    T *d_out = in_tensors[i].get_ptr();
    HCTR_LIB_THROW(cudaMemcpy(h_bprop_out[i], d_out, size * sizeof(T), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(test::compare_array_approx<T>(h_bprop_out[i], h_cpu_bprop_out[i], size,
                                              get_eps<T>(enable_tf32_compute)));  // compare dgrad
  }
}
template <typename T>
void multi_head_attention_layer_test_4d_noT(size_t batch_size, size_t head_num, size_t from_seq_len,
                                            size_t to_seq_len, size_t size_per_head,
                                            bool use_mixed_precision = false,
                                            bool enable_tf32_compute = false) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  size_t out_size = batch_size * head_num * from_seq_len * size_per_head;
  int dims = 4;

  Tensors2<T> in_tensors;
  Tensor2<T> in_tensor;

  buff->reserve({batch_size, head_num, from_seq_len, to_seq_len}, &in_tensor);
  in_tensors.push_back(in_tensor);
  buff->reserve({batch_size, head_num, to_seq_len, size_per_head}, &in_tensor);
  in_tensors.push_back(in_tensor);
  Tensors2<T> out_tensors;

  MultiHeadAttentionLayer multi_head_attention_layer(in_tensors, out_tensors, buff, head_num, false,
                                                     test::get_default_gpu(), use_mixed_precision,
                                                     enable_tf32_compute);

  buff->allocate();

  size_t num = 2;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  for (size_t i = 0; i < 2; i++) {
    h_d_ins[i] = in_tensors[i].get_ptr();
  }

  // test 4d input for mha layer
  for (size_t i = 0; i < num; i++) {
    size_t size = batch_size * head_num * in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    h_ins[i] = new T[size];
    h_bprop_out[i] = new T[size];
    h_cpu_bprop_out[i] = new T[size];
    simulator.fill(h_ins[i], test::align_to_even(size));
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], size * sizeof(T), cudaMemcpyHostToDevice));
  }
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  std::shared_ptr<GPUResource> gpu_resource = test::get_default_gpu();
  cudaStreamSynchronize(gpu_resource->get_stream());
  std::cout << cudaGetLastError() << std::endl;

  T *d_out = out_tensors[0].get_ptr();
  HCTR_LIB_THROW(
      cudaMemcpy(h_out.get(), d_out, out_tensors[0].get_size_in_bytes(), cudaMemcpyDeviceToHost));
  multi_head_attention_cpu_noT(h_ins[0], h_ins[1], h_cpu_out.get(), batch_size, head_num,
                               from_seq_len, to_seq_len, size_per_head);

  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size,
                                            get_eps<T>(enable_tf32_compute)));

  // device bprop
  simulator.fill(h_out.get(), test::align_to_even(out_size));
  HCTR_LIB_THROW(cudaMemcpy(d_out, h_out.get(), out_size * sizeof(T), cudaMemcpyHostToDevice));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  cudaStreamSynchronize(gpu_resource->get_stream());
  std::cout << cudaGetLastError() << std::endl;

  multi_head_attention_dgrad_cpu_noT(h_out.get(), h_ins.get(), h_cpu_bprop_out.get(), batch_size,
                                     head_num, from_seq_len, to_seq_len, size_per_head);
  for (size_t i = 0; i < num; i++) {
    size_t size = batch_size * head_num * in_tensors[i].get_dimensions()[dims - 2] *
                  in_tensors[i].get_dimensions()[dims - 1];
    T *d_out = in_tensors[i].get_ptr();
    HCTR_LIB_THROW(cudaMemcpy(h_bprop_out[i], d_out, size * sizeof(T), cudaMemcpyDeviceToHost));

    ASSERT_TRUE(test::compare_array_approx<T>(h_bprop_out[i], h_cpu_bprop_out[i], size,
                                              get_eps<T>(enable_tf32_compute)));  // compare dgrad
  }
}
template <typename T>
void multi_head_attention_layer_test_3d(size_t batch_size, size_t seq_len, size_t hidden_dim,
                                        size_t head_num, bool use_mixed_precision = false,
                                        bool enable_tf32_compute = false) {
  std::shared_ptr<GeneralBuffer2<CudaAllocator>> buff = GeneralBuffer2<CudaAllocator>::create();

  Tensors2<T> in_3d_tensors;
  Tensor2<T> in_3d_tensor;

  size_t in_size = batch_size * seq_len * hidden_dim;
  size_t out_size = batch_size * head_num * seq_len * seq_len;

  buff->reserve({batch_size, seq_len, hidden_dim}, &in_3d_tensor);
  in_3d_tensors.push_back(in_3d_tensor);
  buff->reserve({batch_size, seq_len, hidden_dim}, &in_3d_tensor);
  in_3d_tensors.push_back(in_3d_tensor);
  buff->reserve({batch_size, seq_len, hidden_dim}, &in_3d_tensor);
  in_3d_tensors.push_back(in_3d_tensor);
  Tensors2<T> out_3d_tensors;

  MultiHeadAttentionLayer multi_head_attention_3d_layer(in_3d_tensors, out_3d_tensors, buff,
                                                        head_num, true, test::get_default_gpu(),
                                                        use_mixed_precision, enable_tf32_compute);
  buff->allocate();

  size_t num = 3;
  std::unique_ptr<T *[]> h_d_ins(new T *[num]);

  std::unique_ptr<T *[]> h_ins(new T *[num]);
  std::unique_ptr<T[]> h_out(new T[out_size]);
  std::unique_ptr<T[]> h_value_out(new T[in_size]);
  std::unique_ptr<T[]> h_cpu_out(new T[out_size]);
  std::unique_ptr<T[]> h_cpu_value_out(new T[in_size]);
  std::unique_ptr<T *[]> h_bprop_out(new T *[num]);
  std::unique_ptr<T *[]> h_cpu_bprop_out(new T *[num]);

  test::GaussianDataSimulator simulator(0.0f, 1.0f);

  // test 3d input for mha layer
  for (size_t i = 0; i < num; i++) {
    h_d_ins[i] = in_3d_tensors[i].get_ptr();
  }
  for (size_t i = 0; i < num; i++) {
    h_ins[i] = new T[in_size];
    h_bprop_out[i] = new T[in_size];
    h_cpu_bprop_out[i] = new T[in_size];
    simulator.fill(h_ins[i], test::align_to_even(in_size));
    HCTR_LIB_THROW(cudaMemcpy(h_d_ins[i], h_ins[i], in_size * sizeof(T), cudaMemcpyHostToDevice));
  }

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_3d_layer.fprop(true);
  HCTR_LIB_THROW(cudaDeviceSynchronize());
  T *d_out = out_3d_tensors[0].get_ptr();
  T *d_value_out = out_3d_tensors[1].get_ptr();

  HCTR_LIB_THROW(cudaMemcpy(h_out.get(), d_out, out_3d_tensors[0].get_size_in_bytes(),
                            cudaMemcpyDeviceToHost));
  HCTR_LIB_THROW(cudaMemcpy(h_value_out.get(), d_value_out, out_3d_tensors[1].get_size_in_bytes(),
                            cudaMemcpyDeviceToHost));
  multi_head_attention_3d_cpu(h_ins[0], h_ins[1], h_ins[2], h_cpu_out.get(), h_cpu_value_out.get(),
                              batch_size, seq_len, hidden_dim, head_num);
  ASSERT_TRUE(test::compare_array_approx<T>(h_out.get(), h_cpu_out.get(), out_size,
                                            get_eps<T>(enable_tf32_compute)));
  ASSERT_TRUE(test::compare_array_approx<T>(h_value_out.get(), h_cpu_value_out.get(), in_size,
                                            get_eps<T>(enable_tf32_compute)));

  HCTR_LIB_THROW(cudaDeviceSynchronize());
  multi_head_attention_3d_layer.bprop();
  HCTR_LIB_THROW(cudaDeviceSynchronize());

  multi_head_attention_dgrad_3d_cpu(h_out.get(), h_value_out.get(), h_ins.get(),
                                    h_cpu_bprop_out.get(), batch_size, head_num, seq_len,
                                    hidden_dim);
  for (size_t i = 0; i < 3; i++) {
    T *d_out = in_3d_tensors[i].get_ptr();
    HCTR_LIB_THROW(cudaMemcpy(h_bprop_out[i], d_out, in_size * sizeof(T), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(test::compare_array_approx<T>(h_bprop_out[i], h_cpu_bprop_out[i], in_size,
                                              get_eps<T>(enable_tf32_compute)));  // compare dgrad
  }
}
}  // namespace

TEST(mha_layer, fp32_512x4x400x600) {
  multi_head_attention_layer_test_4d<float>(512, 4, 400, 600, 128);
}

TEST(mha_layer, tf32_512x4x200x200) {
  multi_head_attention_layer_test_4d<float>(512, 4, 200, 200, 256, false, true);
}

TEST(mha_layer, fp16_512x4x100x200) {
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major < 7) {
    GTEST_SKIP();
  }
  multi_head_attention_layer_test_4d<__half>(512, 4, 100, 200, 256, true, false);
}

TEST(mha_layer, fp32_512x4x400x128) {
  multi_head_attention_layer_test_4d_noT<float>(512, 4, 400, 600, 128);
}

TEST(mha_layer, fp32_512x4x400x128_test) {
  multi_head_attention_layer_test_4d_noT<float>(512, 4, 400, 600, 128);
}
TEST(mha_layer, tf32_512x4x200x256) {
  multi_head_attention_layer_test_4d_noT<float>(512, 4, 200, 200, 256, false, true);
}

TEST(mha_layer, fp32_512x300x128) { multi_head_attention_layer_test_3d<float>(512, 300, 128, 4); }
TEST(mha_layer, fp32_256x100x1024) { multi_head_attention_layer_test_3d<float>(256, 100, 1024, 8); }
TEST(mha_layer, fp32_128x200x256) { multi_head_attention_layer_test_3d<float>(128, 200, 256, 16); }
