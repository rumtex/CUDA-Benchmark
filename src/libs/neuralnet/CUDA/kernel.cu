// CUDA utilities and system includes
#include <cuda_runtime.h>

#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization

#include "libs/neuralnet/CUDA/kernel.h"

#ifndef WIN32
#include <sys/mman.h> // for mmap() / munmap()
#endif

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

inline void
AllocateHostMemory(bool bPinGenericMemory, void **pp_a, void **ppAligned_a, int nbytes)
{
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    if (bPinGenericMemory)
    {
        // allocate a generic page-aligned chunk of system memory
#ifdef WIN32
        // DEBUG_LOG("> VirtualAlloc() allocating %4.2f Mbytes of (generic page-aligned system memory)\n", (float)nbytes/1048576.0f);
        *pp_a = (void *) VirtualAlloc(NULL, (nbytes + MEMORY_ALIGNMENT), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
#else
        // DEBUG_LOG("> mmap() allocating %4.2f Mbytes (generic page-aligned system memory)\n", (float)nbytes/1048576.0f);
        *pp_a = (void *) mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
#endif

        *ppAligned_a = (void *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);

        // DEBUG_LOG("> cudaHostRegister() registering %4.2f Mbytes of generic allocated system memory %p %p\n", (float)nbytes/1048576.0f, ppAligned_a, pp_a);
        // pin allocate memory
        checkCudaErrors(cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped));
    }
    else
#endif
#endif
    {
        // DEBUG_LOG("> cudaMallocHost() allocating %4.2f Mbytes of system memory\n", (float)nbytes/1048576.0f);
        // allocate host memory (pinned is required for achieve asynchronicity)
        checkCudaErrors(cudaMallocHost((void **)pp_a, nbytes));
        *ppAligned_a = *pp_a;
    }
}

inline void
FreeHostMemory(bool bPinGenericMemory, void **pp_a, void **ppAligned_a, int nbytes)
{
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    // CUDA 4.0 support pinning of generic host memory
    if (bPinGenericMemory)
    {
        // unpin and delete host memory
        checkCudaErrors(cudaHostUnregister(*ppAligned_a));
#ifdef WIN32
        VirtualFree(*pp_a, 0, MEM_RELEASE);
#else
        munmap(*pp_a, nbytes);
#endif
    }
    else
#endif
#endif
    {
        cudaFreeHost(*pp_a);
    }
}

#include "libs/neuralnet/CUDA/kernel.cuh"

#if defined(__APPLE__) || defined(MACOSX)
bool bPinGenericMemory = false;  // Generic Pinning of System Paged memory not currently supported on Mac OSX
#else
bool bPinGenericMemory = true;
#endif

extern "C"
void CUDA_set_device(int device_num)
{
    checkCudaErrors(cudaSetDevice(device_num));
}

extern "C"
int CUDA_init_device_generic_host_mem_props(int device_num)
{
    // так быстрее в 1,5 раза Оо
    // checkCudaErrors(cudaSetDevice(device_num));

    // int cuda_device = findCudaDevice(0, NULL);

    // check the compute capability of the device
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    if (0 == num_devices)
    {
        LOG("your system does not have a CUDA capable device, waiving test...\n");
        exit(EXIT_WAIVED);
        return -1;
    }

    // check if the command-line chosen device ID is within range, exit if not
    if (device_num >= num_devices)
    {
        LOG("device_num=%d is invalid, must choose device ID between 0 and %d\n", device_num, num_devices-1);
        exit(EXIT_FAILURE);
        return -1;
    }

    checkCudaErrors(cudaSetDevice(device_num));
    // Checking for compute capabilities
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device_num));

    // Check if GPU can map host memory (Generic Method), if not then we override bPinGenericMemory to be false
    if (bPinGenericMemory)
    {
        LOG("Device: <%s> canMapHostMemory: %s\n", deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");

        if (deviceProp.canMapHostMemory == 0)
        {
            LOG("Using cudaMallocHost, CUDA device does not support mapping of generic host memory\n");
            bPinGenericMemory = false;
        }
    }

    // int device_sync_method = cudaDeviceBlockingSync; // by default we use BlockingSync
    // // enable use of blocking sync, to reduce CPU usage
    // LOG("> Using CPU/GPU Device Synchronization method (%s)\n", sDeviceSyncMethod[device_sync_method]);
    // checkCudaErrors(cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0)));

    // checkCudaErrors(cudaDeviceReset());
    // checkCudaErrors(cudaSetDeviceFlags(0));

    // unsigned int flags = 0;
    // cudaGetDeviceFlags не работает у меня почему-то
    // checkCudaErrors(cudaGetDeviceFlags(&flags));

    // DEBUG_LOG("flags %u\n", flags);
    // if (flags & cudaDeviceScheduleAuto) DEBUG_LOG("device flag cudaDeviceScheduleAuto setted\n");
    // if (flags & cudaDeviceScheduleSpin) DEBUG_LOG("device flag cudaDeviceScheduleSpin setted\n");
    // if (flags & cudaDeviceScheduleYield) DEBUG_LOG("device flag cudaDeviceScheduleYield setted\n");
    // if (flags & cudaDeviceScheduleBlockingSync) DEBUG_LOG("device flag cudaDeviceScheduleBlockingSync setted\n");
    // if (flags & cudaDeviceBlockingSync) DEBUG_LOG("device flag cudaDeviceBlockingSync setted\n");
    // if (flags & cudaDeviceMapHost) DEBUG_LOG("device flag cudaDeviceMapHost setted\n");
    // if (flags & cudaDeviceLmemResizeToMax) DEBUG_LOG("device flag cudaDeviceLmemResizeToMax setted\n");

    return 0;
}

extern "C"
void CUDA_init_run_state(weight* h_train_state, size_t train_bits, cuda_gpu_device_ptrs& ptrs, size_t out_vertex_sum, array<LayerData> &layers) {
    cudaStream_t* stream = (cudaStream_t*) malloc(sizeof(cudaStream_t));
    checkCudaErrors(cudaStreamCreate((cudaStream_t*) stream));
    ptrs.stream = stream;

    checkCudaErrors(cudaMalloc((void**)&ptrs.d_train_state, train_bits * float_size));
    checkCudaErrors(cudaMalloc((void**)&ptrs.d_float_train_state, train_bits * float_size));
    checkCudaErrors(cudaMalloc((void**)&ptrs.d_voters_volume, out_vertex_sum * float_size));
}

extern "C"
void CUDA_update_run_state(weight* h_train_state, size_t train_bits, cuda_gpu_device_ptrs& ptrs, size_t out_vertex_sum, array<LayerData> &layers) {
    cudaStream_t* stream = (cudaStream_t*) ptrs.stream;

    checkCudaErrors(cudaMemcpy(ptrs.d_train_state, h_train_state, train_bits, cudaMemcpyHostToDevice));

    dim3 grid_train(train_bits / 4, 4);
    dim3 blocks(1,1);

    wtf<<<grid_train, blocks, 0, *stream>>>(ptrs.d_train_state, ptrs.d_float_train_state);

    float* d_voters_volume_ptr = ptrs.d_voters_volume;
    fweight* d_float_train_state_ptr = ptrs.d_float_train_state;
    for (auto layer : layers) {
        dim3 grid_voters(layer->out_vertex_count, 1);
        cvv<<<grid_voters, blocks, 0, *stream>>>(d_voters_volume_ptr, d_float_train_state_ptr, layer->vertex_count);
        d_voters_volume_ptr += layer->out_vertex_count;
        d_float_train_state_ptr += layer->vertex_count * layer->out_vertex_count;
    }

    checkCudaErrors(cudaStreamSynchronize(*stream));
}

extern "C"
void CUDA_update_train_byte_async(cuda_gpu_device_ptrs& ptrs, size_t bit_num, float* bit_value, LayerData& layer, size_t& before_out_vertexes_sum, int& value_num, float* new_vote_sum) {
    cudaStream_t* stream = (cudaStream_t*) ptrs.stream;

    checkCudaErrors(cudaMemcpyAsync(((float*)ptrs.d_float_train_state+bit_num+value_num), bit_value, float_size, cudaMemcpyHostToDevice, *stream));

    if (value_num == 3) {
        // A coefficient
        // size_t weight_layer_begin_diff = (bit_num/4 - (layer.weights_ptr - ptrs.h_trained_bit_pool)) % layer.out_vertex_count;
        // size_t weight_layers_begin_diff = weight_layer_begin_diff + (layer.weights_ptr - ptrs.h_trained_bit_pool);
        // DEBUG_LOG("%p + %zu: %p set %f\n", ptrs.d_voters_volume, before_out_vertexes_sum, ptrs.d_voters_volume + before_out_vertexes_sum, *new_vote_sum);
        checkCudaErrors(cudaMemcpyAsync(ptrs.d_voters_volume + before_out_vertexes_sum, new_vote_sum, float_size, cudaMemcpyHostToDevice, *stream));
        // cvv_one<<<dim3(1,1), dim3(1,1), 0, stream[0]>>>(
        //     ptrs.d_voters_volume + before_out_vertexes_sum + weight_layer_begin_diff,
        //     (float*)(ptrs.d_float_train_state + weight_layers_begin_diff) + value_num, // +3
        //     layer.vertex_count,
        //     layer.out_vertex_count * float_size
        // );
    }

}

extern "C"
void CUDA_alloc_work_state(size_t work_bits, float*& h_work_state) {
    // Allocate Host memory (could be using cudaMallocHost or VirtualAlloc/mmap if using the new CUDA 4.0 features
    // AllocateHostMemory(bPinGenericMemory, (void**)&haa_work_state, (void**)&h_work_state, work_bits);
    h_work_state = (float*)malloc(work_bits);
}

extern "C"
void CUDA_clear_run_data(cuda_gpu_device_ptrs& ptrs)
{
    checkCudaErrors(cudaFree(ptrs.d_train_state));
    checkCudaErrors(cudaFree(ptrs.d_float_train_state));
    checkCudaErrors(cudaFree(ptrs.d_voters_volume));

    // Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
    // FreeHostMemory(bPinGenericMemory, (void**)&haa_work_state, (void**)&h_work_state, work_bits);
    free(ptrs.h_work_state);
    free(ptrs.stream);
}

extern "C"
void CUDA_RUN(array<LayerData> &layers, cuda_gpu_device_ptrs& ptrs, size_t work_bits, float* input, float* output)
{
    // float time_kernel, time_memcpy;
    cudaStream_t* stream = (cudaStream_t*)ptrs.stream;

    // allocate device memory
    float *d_work_state = 0;
    checkCudaErrors(cudaMalloc((void**)&d_work_state, work_bits));
    // checkCudaErrors(cudaMemset(d_work_state, 0x0, work_bits)); // 0.0f != 0

    checkCudaErrors(cudaMemcpy(d_work_state, input, layers.data()[0].vertex_count * float_size, cudaMemcpyHostToDevice));

    fweight* d_float_train_state_ptr = ptrs.d_float_train_state;
    float* d_voters_volume_ptr = ptrs.d_voters_volume;
    float* d_work_state_ptr = d_work_state;

    for (auto layer : layers) {
        dim3 grid = dim3(layer->out_vertex_count, 1);
        // dim3 blocks = dim3(1, 1);


        run_layer_perception_stage_1<<<grid, 0, 0, *stream>>>(d_work_state_ptr, d_work_state_ptr+layer->vertex_count, d_float_train_state_ptr, 0);
        d_work_state_ptr += layer->vertex_count;

        run_layer_perception_stage_2<<<layer->out_vertex_count, 0, 0, *stream>>>(d_work_state_ptr, d_float_train_state_ptr, d_voters_volume_ptr, 0);
        // run_perception<<<grid, blocks, 0, *stream>>>(d_work_state_ptr, d_work_state_ptr+layer->vertex_count, d_float_train_state_ptr, d_voters_volume_ptr, layer->vertex_count, 0);

        d_voters_volume_ptr += layer->out_vertex_count;
        d_float_train_state_ptr += layer->vertex_count * layer->out_vertex_count;
    }
    checkCudaErrors(cudaMemcpyAsync(ptrs.h_work_state, d_work_state, work_bits, cudaMemcpyDeviceToHost, *stream));

    size_t output_size = layers.data()[layers.size() - 1].out_vertex_count;
    size_t exc_out_vertexes_count = work_bits / 4 - output_size;

    checkCudaErrors(cudaStreamSynchronize(*stream));

    for (int i = 0; i < output_size; i++) {
        output[i] = ptrs.h_work_state[exc_out_vertexes_count + i];
        // printf("result %i:\t%.3f\n", i, ptrs.h_work_state[(work_bits/4 - output_size) + i]);
    }

    checkCudaErrors(cudaFree(d_work_state));
}

float fzero = 0.;

// вместо стримов выполняет несколько запусков (по всей тренировочной выборке) в одном стриме - так быстрее
extern "C"
void CUDA_INIT_STREAMS(size_t nstreams, unsigned char* receptions_bit_pool, float*& d_work_state, size_t work_bytes, float*& h_input, size_t input_size, float*& h_output, size_t output_size) {
    // cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nstreams);
    h_output = (float*)malloc(output_size * nstreams * float_size);
    h_input = (float*)malloc((input_size * nstreams + (work_bytes - input_size)) * float_size);
    // AllocateHostMemory(bPinGenericMemory, (void**)&ha_input, (void**)&h_input, input_size * nstreams * float_size);

    checkCudaErrors(cudaMalloc((void**)&d_work_state, work_bytes * nstreams * float_size));
    // DEBUG_LOG("work_bytes: %zu, work_state %p, total: %zu\n", work_bytes, d_work_state,work_bytes * nstreams * float_size);

    unsigned char* bytes_ptr = receptions_bit_pool;
    float* h_input_ptr = h_input;
    for (size_t i = 0; i < nstreams; i++)
    {
        for (size_t bytes_it = 0; bytes_it < input_size; bytes_it++) {
            h_input_ptr[bytes_it] = bytes_ptr[bytes_it] / 255.;
        }
        bytes_ptr+=input_size + output_size;
        h_input_ptr+=input_size;

        // checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }
    for (size_t i = 0; i < (work_bytes - input_size); i++, h_input_ptr++) {
        *h_input_ptr = fzero;
    }

    // streams_ptr = (void*)streams;
}

extern "C"
void CUDA_CLEAR_STREAMS(size_t nstreams, float*& d_work_state, size_t work_bytes, float*& h_input, size_t input_size, float*& h_output, size_t output_size) {
    // cudaStream_t* streams = (cudaStream_t*) streams_ptr;

    // for (size_t i = 0; i < nstreams; i++)
    // {
    //     checkCudaErrors(cudaStreamDestroy(streams[i]));
    // }

    // checkCudaErrors(cudaHostUnregister(h_output));
    checkCudaErrors(cudaFree(d_work_state));
    // free(streams);

    // checkCudaErrors(cudaFreeHost(h_output));
    free(h_output);
    free(h_input);
    // FreeHostMemory(bPinGenericMemory, (void**)&ha_input, (void**)&h_input, input_size * nstreams * float_size);
}

extern "C"
float CUDA_ACCURACY_CHECK(array<LayerData> &layers, cuda_gpu_device_ptrs& ptrs, unsigned char* receptions_bit_pool, float* d_work_state, size_t work_bytes, float* h_input, size_t input_size, float* h_output, size_t output_size, size_t nstreams) {
    cudaStream_t* stream = (cudaStream_t*) ptrs.stream;
    float accuracy = 0.;

    float* h_input_ptr = h_input;
    float* d_work_state_ptr = d_work_state;
    float* zero_float_work_state = h_input+(input_size * nstreams);
    for (size_t i = 0; i < nstreams; i++)
    {
        checkCudaErrors(cudaMemcpyAsync(d_work_state_ptr, h_input_ptr, input_size * float_size, cudaMemcpyHostToDevice, *stream));

        //TODO с девайса на девайс больше GB/s
        // checkCudaErrors(cudaMemsetAsync(d_work_state_ptr + input_size, zero, (work_bytes - input_size)*float_size, *stream));
        checkCudaErrors(cudaMemcpyAsync(d_work_state_ptr + input_size, zero_float_work_state, (work_bytes - input_size)*float_size, cudaMemcpyHostToDevice, *stream));
        d_work_state_ptr += work_bytes;
        h_input_ptr += input_size;
    }

    fweight* d_float_train_state_ptr;
    float* d_voters_volume_ptr;
    d_work_state_ptr = d_work_state;
    size_t output_counter = 0;
    d_float_train_state_ptr = ptrs.d_float_train_state;
    d_voters_volume_ptr = ptrs.d_voters_volume;
    d_work_state_ptr = d_work_state;

    for (auto layer : layers) {
        dim3 grid = dim3(layer->vertex_count, layer->out_vertex_count);

        run_layer_perception_stage_1<<<grid, nstreams, 0, *stream>>>(d_work_state_ptr, d_work_state_ptr+layer->vertex_count, d_float_train_state_ptr, work_bytes);
        d_work_state_ptr += layer->vertex_count;

        run_layer_perception_stage_2<<<layer->out_vertex_count, nstreams, 0, *stream>>>(d_work_state_ptr, d_float_train_state_ptr, d_voters_volume_ptr, work_bytes);
        // run_perception<<<grid, blocks, 0, *stream>>>(d_work_state_ptr, d_work_state_ptr+layer->vertex_count, d_float_train_state_ptr, d_voters_volume_ptr, layer->vertex_count, work_bytes);

        d_voters_volume_ptr += layer->out_vertex_count;
        d_float_train_state_ptr += layer->vertex_count * layer->out_vertex_count;
    }

    for (size_t i = 0; i < nstreams; i++)
    {
        checkCudaErrors(cudaMemcpyAsync(h_output + output_counter, d_work_state_ptr + i * work_bytes, output_size * float_size, cudaMemcpyDeviceToHost, *stream));
        output_counter += output_size;
    }

    unsigned char* bytes_ptr = receptions_bit_pool;
    float* output_ptr = h_output;
    checkCudaErrors(cudaStreamSynchronize(*stream));
    for (size_t i = 0; i < nstreams; i++)
    {
        float run_accuracy = 0.;
        for (size_t bytes_it = 0; bytes_it < output_size; bytes_it++) {
            // DEBUG_LOG("accuracy %.15f %i ~ %.15f\n", output_ptr[bytes_it], (bytes_ptr[input_size + bytes_it] ? 1 : 0), 1.0 - mod(output_ptr[bytes_it] - (bytes_ptr[input_size + bytes_it] ? 1. : 0.)));
            if ((output_ptr[bytes_it] <= 0.5f && bytes_ptr[input_size + bytes_it] == 255)
                || (output_ptr[bytes_it] > 0.5f && bytes_ptr[input_size + bytes_it] == 0)) {
                continue;
                // run_accuracy = 0.;
                // break;
            }
            run_accuracy += 1.0 - mod(output_ptr[bytes_it] - bytes_ptr[input_size + bytes_it] / 255.);

            // if (isnan(accuracy)) {
            //     float* work_state = new float[work_bytes];
            //     checkCudaErrors(cudaMemcpy(work_state, d_work_state + i * work_bytes, work_bytes * float_size, cudaMemcpyDeviceToHost));

            //     for(size_t work_bytes_it = 0; work_bytes_it < work_bytes; work_bytes_it++) {
            //         DEBUG_LOG("byte[%zu] = %f\n", work_bytes_it, work_state[work_bytes_it]);
            //     }

            //     DEBUG_LOG("output_byte[%zu]: %f, (wait for %i)\n", bytes_it, output_ptr[bytes_it], bytes_ptr[input_size + bytes_it] ? 1 : 0);
            //     exit(EXIT_FAILURE);
            // }
        }
        accuracy += run_accuracy;

        bytes_ptr += input_size + output_size;
        output_ptr += output_size;
    }

    accuracy /= output_size * nstreams;
    return accuracy;
}


