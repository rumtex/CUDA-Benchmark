
const char *sEventSyncMethod[] =
{
    "cudaEventDefault",
    "cudaEventBlockingSync",
    "cudaEventDisableTiming",
    NULL
};

const char *sDeviceSyncMethod[] =
{
    "cudaDeviceScheduleAuto",
    "cudaDeviceScheduleSpin",
    "cudaDeviceScheduleYield",
    "INVALID",
    "cudaDeviceScheduleBlockingSync",
    NULL
};


__global__ void run_layer_perception_stage_1(float* in_layer, float* out_layer, fweight* trained_state, size_t work_bytes)
{
    size_t work_bytes_offset = threadIdx.x * work_bytes; // параллельный запуск на нескольких исходных данных

    float &out_vertex = out_layer[work_bytes_offset + blockIdx.y];
    float &in_vertex = in_layer[work_bytes_offset + blockIdx.x];

    fweight* vec = &trained_state[blockIdx.x * gridDim.y + blockIdx.y];

    // if (threadIdx.x == 0) printf("input_vertex[%d/%d], in: %f, weight(r: %f, g: %f, b: %f, a: %f), out: %f, %p\n", blockIdx.x+1,gridDim.x, in_vertex, vec->r, vec->g, vec->b, vec->a, out_vertex, &out_vertex);

    float vote;
    if (vec->r < 0.0) {
        vote = -((powf(-vec->r, 1.3) * in_vertex) + (1-in_vertex)) + 1;
    } else
    if (vec->r == 0.0) {
        vote = in_vertex;
    } else {
        vote = (powf(vec->r, 1.3) * (1-in_vertex) + in_vertex);
    }

    // if (threadIdx.x == 0) printf("stage 1 after %f r = %f vote: %f\n", in_vertex, vec->r, vote);

    if (vec->g > 0.) vote += vec->g * (0.5 - vote)/0.5;

    atomicAdd(&out_vertex, vote * vec->a);
    // if (threadIdx.x == 0) printf("blockIdx(x: %d, y: %d, z: %d)(%d;%d;%d)\tthreadIdx(x: %d, y: %d, z: %d)(%d;%d;%d)\n", blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
}

__global__ void run_layer_perception_stage_2(float* out_layer, fweight* trained_state, float* d_voters_volume, size_t work_bytes)
{
    size_t work_bytes_offset = threadIdx.x * work_bytes; // параллельный запуск на нескольких исходных данных

    float *vertex = &out_layer[work_bytes_offset + blockIdx.x];

    fweight* vec = &trained_state[blockIdx.x];

    // if (threadIdx.x == 0) printf("stage 2 sum %f/%f = %f\n", *vertex, d_voters_volume[blockIdx.x], *vertex / d_voters_volume[blockIdx.x]);

    *vertex /= d_voters_volume[blockIdx.x];

    if (vec->g < 0.) {
        // if (threadIdx.x == 0) printf("%f XOR %f = %f\n", *vertex, vec->g, fabsf(*vertex + vec->g));
        *vertex = fabsf(*vertex + vec->g);
    }
    // if (threadIdx.x == 0) printf("blockIdx(x: %d, y: %d, z: %d)(%d;%d;%d)\tthreadIdx(x: %d, y: %d, z: %d)(%d;%d;%d)\n", blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
}

__global__ void run_perception(float* in_layer, float* out_layer, fweight* trained_state, float* d_voters_volume, size_t in_count, size_t work_bytes)
{
    float& vertex = out_layer[blockIdx.x + blockIdx.y * work_bytes]; //  + blockIdx.y * work_bytes - параллельный запуск
    vertex = 0.0;
    fweight* vec;// = &trained_state[in_count * blockIdx.x];

    vec = &trained_state[blockIdx.x];
    for (size_t i=0; i < in_count; i++)
    {
        size_t it = i + blockIdx.y * work_bytes;

        // if (blockIdx.y == 0) printf("input_vertex[%lu/%lu], gridDim.x(%d): %f, weight(r: %f, g: %f, b: %f, a: %f) now: %f, %p\n", i+1, in_count, gridDim.x, in_layer[it], vec->r, vec->g, vec->b, vec->a, vertex, &vertex);

        float vote;
        if (vec->r < 0.0) {
            vote = -((powf(-vec->r, 1.3) * in_layer[it]) + (1-in_layer[it])) + 1;
        } else
        if (vec->r == 0.0) {
            vote = in_layer[it];
        } else {
            vote = (powf(vec->r, 1.3) * (1-in_layer[it]) + in_layer[it]);
        }

        // if (blockIdx.y == 0) printf("stage 1 after %f r = %f vote: %f/%f\n", in_layer[it], vec->r, vote, d_voters_volume[blockIdx.x]);

        if (vec->g > 0.) vote += vec->g * (0.5 - vote)/0.5;

        vertex += vote * vec->a;

        vec += gridDim.x;
    }

    vertex /= d_voters_volume[blockIdx.x];

    vec = &trained_state[blockIdx.x];
    for (size_t i=0; i < in_count; i++)
    {
        if (vec->g < 0.) {
            // if (blockIdx.y == 0) printf("%f XOR %f = %f\n", vertex, vec->g, fabsf(vertex + vec->g));
            vertex = fabsf(vertex + vec->g);
        }

        vec += gridDim.x;
    }

    // if (blockIdx.y == 0) printf("blockIdx(x: %d, y: %d, z: %d),\tthreadIdx(x: %d, y: %d, z: %d)\tidx: %d = %f\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, vertex);

}

__device__ float d_char_to_float(unsigned char a) {
    return a & MINUS_SIGN_MASK ? (a - 128) / -127.f : a / 127.f;
}

__global__ void wtf(weight* trained_state, fweight* f_trained_state)
{
    unsigned char* trained_byte = (unsigned char*)&trained_state[blockIdx.x];
    ((float*)&f_trained_state[blockIdx.x])[blockIdx.y] = (blockIdx.y == 3) ? trained_byte[blockIdx.y] / 255.f : d_char_to_float(trained_byte[blockIdx.y]);

    // printf("blockIdx(x: %d, y: %d, z: %d),\tthreadIdx(x: %d, y: %d, z: %d)\tbyte: %d[%d] = %hu, in float: %f \n",
    //     blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, trained_byte[blockIdx.y], ((float*)&f_trained_state[blockIdx.x])[blockIdx.y]);

}

__global__ void cvv(float* d_voters_volume, fweight* d_float_train_state, size_t input_vertex_count) {
    d_voters_volume[blockIdx.x] = 0.f;

    for (size_t i = 0; i < input_vertex_count; i++) {
        d_voters_volume[blockIdx.x] += ((float*)&d_float_train_state[blockIdx.x + gridDim.x * i])[3];
    }

    // printf("cvv blockIdx(x: %d, y: %d, z: %d),\tthreadIdx(x: %d, y: %d, z: %d)\tbyte: %d/%d (%lu vetrexes) =  %f \n",
    //     blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x + 1, blockDim.y, input_vertex_count, d_voters_volume[blockIdx.x]);

}

// __global__ void cvv_one(float* d_voters_volume_ptr, float* d_float_train_state_ptr, size_t input_vertex_count, size_t output_vertex_size) {
//     d_voters_volume_ptr[0] = 0.f;

//     for (size_t i = 0; i < input_vertex_count; i++) {
//         d_voters_volume_ptr[0] += d_float_train_state_ptr[output_vertex_size * i];
//         // printf("%p: %f+%f\n", &d_float_train_state_ptr[output_vertex_size * i], d_float_train_state_ptr[output_vertex_size * i], d_voters_volume_ptr[0]);
//     }

//     // printf("cvv blockIdx(x: %d, y: %d, z: %d),\tthreadIdx(x: %d, y: %d, z: %d)\tbyte: %d/%d (%lu vetrexes) =  %f \n",
//     //     blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x + 1, blockDim.y, input_vertex_count, d_voters_volume_ptr[0]);

// }
