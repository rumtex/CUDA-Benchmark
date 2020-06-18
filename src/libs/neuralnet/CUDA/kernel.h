#ifndef __CUDA_KERNEL_H_
#define __CUDA_KERNEL_H_

#include "libs/neuralnet/Perceptron.h"

extern "C" int CUDA_init_device_generic_host_mem_props(int device_num);

extern "C" void CUDA_init_run_state(weight* h_train_state, size_t train_bits, cuda_gpu_device_ptrs& ptrs, size_t out_vertex_sum, array<LayerData> &layers);
extern "C" void CUDA_update_run_state(weight* h_train_state, size_t train_bits, cuda_gpu_device_ptrs& ptrs, size_t out_vertex_sum, array<LayerData> &layers);

extern "C" void CUDA_update_train_byte_async(cuda_gpu_device_ptrs& ptrs, size_t byte_num, float* byte_value, LayerData& layer, size_t& before_vertexes_sum, int& value_num, float* new_vote_sum);

extern "C" void CUDA_alloc_work_state(size_t work_bits, float*& h_work_state);
extern "C" void CUDA_set_device(int device_num);
extern "C" void CUDA_clear_run_data(cuda_gpu_device_ptrs& ptrs);

extern "C" void CUDA_RUN(array<LayerData> &layers, cuda_gpu_device_ptrs& ptrs, size_t work_bits, float* input, float* output);

extern "C" void CUDA_INIT_STREAMS(size_t nstreams, unsigned char* receptions_bit_pool, float*& d_work_state, size_t work_bytes, float*& h_input, size_t input_size, float*& h_output, size_t output_size);
extern "C" void CUDA_CLEAR_STREAMS(size_t nstreams, float*& d_work_state, size_t work_bytes, float*& h_input, size_t input_size, float*& h_output, size_t output_size);

extern "C" float CUDA_ACCURACY_CHECK(array<LayerData> &layers, cuda_gpu_device_ptrs& ptrs, unsigned char* receptions_bit_pool, float* d_work_state, size_t work_bytes, float* h_input, size_t input_size, float* h_output, size_t output_size, size_t nstreams);

#endif //__CUDA_KERNEL_H_
