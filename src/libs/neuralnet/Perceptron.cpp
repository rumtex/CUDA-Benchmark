#include <libs/neuralnet/Perceptron.h>

#include "libs/neuralnet/CUDA/kernel.h"

size_t calc_ways(PerceptronConfiguration& p_config) {
    size_t ways = 1;
    for (auto layer_size : p_config.layer_sizes) {
        ways *= *layer_size;
    }
    return ways;
}

Perceptron::Perceptron(PerceptronConfiguration p_init_config) :
    p_config{
        // memory leak fix нынче так выглядит
        .name = p_init_config.name,
        .layer_sizes = (p_init_config.layer_sizes.size()),
        .mode = p_init_config.mode,
        .log_to_json = p_init_config.log_to_json,
        .gpu_num = p_init_config.gpu_num,
    },
    p_data{
        .ways = {calc_ways(p_init_config)},
        .layers = (p_init_config.layer_sizes.size() - 1)
    }
{
    p_data.cuda_ptrs.num = -1;

    // reserve mem blocks
    {
        size_t prev_layer_size = 0;

        // p_config.layer_sizes = {2,1};//p_init_config.layer_sizes;

        DEBUG_LOG("initialize perceptron configuration...\n"
                "layers count: %zu\n"
                "layers size: ", p_config.layer_sizes.size());

        size_t layer_sizes_it = 0;
        p_data.out_vertex_sum = 0;
        for (auto layer_size : p_init_config.layer_sizes) {
            if (layer_sizes_it != p_config.layer_sizes.size() - 1)
                p_data.layers.data()[layer_sizes_it].vertex_count = *layer_size;
            if (layer_sizes_it != 0) {
                p_data.layers.data()[layer_sizes_it - 1].out_vertex_count = *layer_size;
                p_data.out_vertex_sum += *layer_size;
            }
            if (*layer_size <= 0) throw "Perceptron config: count vertexes must be positive";
            p_config.layer_sizes.data()[layer_sizes_it++] = *layer_size;
            if (*layer_size > p_data.max_layer_size) p_data.max_layer_size = *layer_size;
            p_data.working_bit_pool_size += *layer_size * float_size;
            if (prev_layer_size != 0 )
                p_data.trained_bytes_count += *layer_size * prev_layer_size;
            prev_layer_size = *layer_size;
            DEBUG_LOG("%zu ", *layer_size);
        }

        p_data.voters_volume = (float*) calloc(p_data.out_vertex_sum, sizeof(float));
        p_data.layers.data()[0].voters_volume = p_data.voters_volume;

        p_data.trained_bit_pool_size = p_data.trained_bytes_count * float_size;

        DEBUG_LOG("\n"
            "trained_bit_pool_size: %zu\n"
            "working_bit_pool_size: %zu\n"
            "weight vectors: %zu\n", p_data.trained_bit_pool_size, p_data.working_bit_pool_size, p_data.ways.size());

        p_data.input_size = **p_config.layer_sizes.begin();
        p_data.output_size = prev_layer_size;
        p_data.trained_bit_pool = (weight*) calloc(p_data.trained_bit_pool_size, 1);
        p_data.float_trained_bit_pool = (fweight*) calloc(p_data.trained_bit_pool_size, float_size);
    }

    // name && saved condition file
    {
        int name_len = strlen(p_config.name);
        if (name_len == 0) throw "Perceptron config: invalid name length";

        constexpr char ext[] = ".fm";
        p_data.filename = new char[name_len + strlen(ext) + 1];
        std::memcpy(p_data.filename, p_config.name, name_len);
        std::memcpy(p_data.filename+name_len, ext, strlen(ext) + 1);
        // DEBUG_LOG("filename: \"%s\"\n", p_data.filename);

        int config_fd = open(p_data.filename
        , O_RDWR | O_CREAT
        , S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

        if (config_fd == -1) throw (std::string("Perceptron config: cant open file ") + p_data.filename).c_str();

        ssize_t cv = read(config_fd, &p_data.train_it_count, sizeof(size_t));
        // DEBUG_LOG("Load %zu from file\n", cv);

        if (cv == sizeof(size_t)) {
            cv += pread(config_fd, p_data.trained_bit_pool, p_data.trained_bit_pool_size, cv);
            // DEBUG_LOG("Load %zu from file\n", cv);
        } else {
            cv = -1;
        }

        if (cv == sizeof(size_t) + p_data.trained_bit_pool_size) {
            if (!is_valid_trained_bits()) cv = -1;
            if (cv != -1) {
                ssize_t recept_bits_count;
                if (false) {
                    // TODO когда может быть не предусмотрено хранение рецептов тренировки
                    recept_bits_count = 0;
                    cv = 0;
                } else {
                    recept_bits_count = (p_data.input_size + p_data.output_size) * p_data.train_it_count;
                    p_data.receptions_bit_pool = (unsigned char*)calloc(recept_bits_count, 1);
                    cv = pread(config_fd, p_data.receptions_bit_pool, recept_bits_count+1, cv);
                    // DEBUG_LOG("Load %zu from file\n", cv);
                }

                if (cv == recept_bits_count) {
                    LOG("Load trained state from file: %s\n", p_data.filename);
                    LOG("trained by %zu/%e receptions v%f\n", p_data.train_it_count, pow(p_data.input_size, 2), p_data.train_it_count / pow(p_data.input_size, 2));
                } else {
                    cv = -1;
                    free(p_data.receptions_bit_pool);
                }
            }
        } else {
            cv = -1;
        }
        close(config_fd);
        unlink(p_data.filename);

        if (cv == -1) {
            LOG("Initialize trained state\n");
            p_data.train_it_count = 0;
            p_data.receptions_bit_pool = (unsigned char*)calloc(0, 1);
            size_t it = 0;
            weight* ptr;
            while (it < p_data.trained_bit_pool_size) {
                ptr = p_data.trained_bit_pool + (it/float_size);
                *ptr = weight(.0,.0,0.,1.);

                // DEBUG_LOG("it: %zu, num %hhu,%hhu,%hhu,%hhu\n", it, ptr->r, ptr->g, ptr->b, ptr->a);
                it += float_size;
            }
            // if (!is_valid_trained_bits()) throw "Bad generating float function";
        }
    }

    // init float tbp
    {
        size_t it = 0;
        weight* tbp_ptr = p_data.trained_bit_pool;
        fweight* float_tbp_ptr = p_data.float_trained_bit_pool;

        while (it < p_data.trained_bit_pool_size) {
            tbp_ptr = p_data.trained_bit_pool + (it/float_size);
            *float_tbp_ptr = fweight(
                char_to_float(tbp_ptr->r),
                char_to_float(tbp_ptr->g),
                char_to_float(tbp_ptr->b),
                tbp_ptr->a / 255.f
            );

            // DEBUG_LOG("it: %zu, num %hhu,%hhu,%hhu,%hhu, float %f,%f,%f,%f\n", it, tbp_ptr->r, tbp_ptr->g, tbp_ptr->b, tbp_ptr->a, float_tbp_ptr->r, float_tbp_ptr->g, float_tbp_ptr->b, float_tbp_ptr->a);
            float_tbp_ptr++;
            it += float_size;
        }
    }

    // ways
    {
        size_t way_length = p_config.layer_sizes.size() - 1;
        size_t vertex_it[p_config.layer_sizes.size()];
        for (size_t it = 0; it < p_config.layer_sizes.size(); it++) {
            vertex_it[it] = 0;
        }

        weight* layers_ptrs[way_length];
        fweight* tbp_ptr = p_data.float_trained_bit_pool;
        for (size_t it = 0; it < way_length; it++) {
            layers_ptrs[it] = (it == 0 ? p_data.trained_bit_pool : layers_ptrs[it - 1] + p_config.layer_sizes.data()[it] * p_config.layer_sizes.data()[it - 1]);
            p_data.layers.data()[it].weights_ptr = layers_ptrs[it];
            p_data.layers.data()[it].ptr = tbp_ptr;

            tbp_ptr += p_data.layers.data()[it].vertex_count * p_data.layers.data()[it].out_vertex_count;
        }

        for (auto way : p_data.ways) {
            way->edges = *(new array<edge>(way_length));
            size_t layer_it = 0;
            // DEBUG_LOG("way: ");
            for (auto edge : way->edges) {
                edge->weight_ptr = (weight*) layers_ptrs[layer_it] + vertex_it[layer_it]*p_config.layer_sizes.data()[layer_it+1] + vertex_it[layer_it + 1];
                edge->in_num = vertex_it[layer_it];
                edge->out_num = vertex_it[layer_it + 1];

                // DEBUG_LOG("layer: %zi (size: %zu), vertex_from: %zu, vertex_to: %zu, prepend %zu, w: %f\n",
                //     layer_it, p_config.layer_sizes.data()[layer_it],
                //     vertex_it[layer_it], vertex_it[layer_it + 1],
                //     vertex_it[layer_it]*p_config.layer_sizes.data()[layer_it+1], *edge->weight_ptr);
                layer_it++;
            }
            // DEBUG_LOG("\n");

            way->in_num = vertex_it[0];
            way->out_num = vertex_it[layer_it];
            while (layer_it >= 0) {
                // DEBUG_LOG("l:%zu, %zu == %zu\n", layer_it, vertex_it[layer_it], p_config.layer_sizes.data()[layer_it] - 1);
                if (vertex_it[layer_it] == p_config.layer_sizes.data()[layer_it] - 1) {
                    vertex_it[layer_it] = 0;
                } else {
                    vertex_it[layer_it]++;
                    break;
                }
                layer_it--;
            }
        }
    }

    if (p_config.gpu_num != -1)
        init_gpu_device(p_config.gpu_num);


    init_run_vars();
}

Perceptron::~Perceptron()
{
    DEBUG_LOG("destroy perceptron %s\n", p_config.name);

    if (p_data.cuda_ptrs.num != -1) {
        clear_gpu_device_mem();
    }

    save_trained_state();
}

static const char red[]    = {0x1b, '[', '1', ';', '3', '1', 'm', 0};
static const char green[]  = {0x1b, '[', '1', ';', '3', '2', 'm', 0};
static const char yellow[] = {0x1b, '[', '1', ';', '3', '3', 'm', 0};
static const char blue[]   = {0x1b, '[', '1', ';', '3', '4', 'm', 0};
static const char normal[] = {0x1b, '[', '0', ';', '3', '9', 'm', 0};
void Perceptron::run(array<bool> input_data, bool* result) {
    DEBUG_LOG("running..\ninput: ");
    for (auto byte : input_data) {
        DEBUG_LOG("%i ", *byte);
    }
    DEBUG_LOG("\n");

    if (p_data.input_size != input_data.size()) throw "bad input bytes count";

    float input[input_data.size()];
    float output[p_data.output_size];
    size_t input_size = 0;
    for (auto byte : input_data) {
        input[input_size++] = *byte ? 1.0 : 0.0;
    }

    run(input, output);

    for (size_t out_vertex_it = 0; out_vertex_it < p_data.output_size; out_vertex_it++) {
        result[out_vertex_it] = (output[out_vertex_it] > 0.5 ? true : false);
        // accuracy += result[out_vertex_it] ? output[out_vertex_it] : 1.0 - output[out_vertex_it];
        DEBUG_LOG("%zu. result bit %s%f%s, round for %i\n", out_vertex_it, green, output[out_vertex_it], normal, (output[out_vertex_it] > 0.5 ? 1 : 0));
    }

}

// TODO с char не релиз, думаю
void Perceptron::run(array<unsigned char> input_data, unsigned char* result) {
    if (p_data.input_size != input_data.size()) throw "bad input bytes count";

    float input[input_data.size()];
    float output[p_data.output_size];
    size_t input_size = 0;
    for (auto byte : input_data) {
        input[input_size++] = *byte / 255.;
    }
    run(input, output);

    for (size_t out_vertex_it = 0; out_vertex_it < p_data.output_size; out_vertex_it++) {
        result[out_vertex_it] = (output[out_vertex_it] * 255);
        // accuracy += result[out_vertex_it] ? output[out_vertex_it] : 1.0 - output[out_vertex_it];
        DEBUG_LOG("%zu. result bit %s%f%s, round for %i\n", out_vertex_it, green, output[out_vertex_it], normal, (output[out_vertex_it] > 0.5 ? 1 : 0));
    }

}

void Perceptron::init_run_vars() {
    size_t out_vertex_counter = 0;
    for (auto layer : p_data.layers) {
        layer->voters_volume = p_data.voters_volume + out_vertex_counter;
        // создаем массив для общего объема голосов
        out_vertex_counter += layer->out_vertex_count;
    }
    update_run_vars();
}

void Perceptron::update_run_vars() {
    if (p_data.cuda_ptrs.num != -1) {
        update_gpu_run_state();
    }

    weight* t_ptr = p_data.trained_bit_pool;
    for (auto layer : p_data.layers) {

        for (size_t out_vertex_it = 0; out_vertex_it < layer->out_vertex_count; out_vertex_it++) {
            layer->voters_volume[out_vertex_it] = 0.;
        }
        for (size_t cur_vertex_it = 0; cur_vertex_it < layer->vertex_count; cur_vertex_it++) {
            for (size_t out_vertex_it = 0; out_vertex_it < layer->out_vertex_count; out_vertex_it++) {
                layer->voters_volume[out_vertex_it] += t_ptr[out_vertex_it].a / 255.0f;
            }
            t_ptr += layer->out_vertex_count;
        }
    }
}

void Perceptron::debug_log_train_state() {
    // log state
    {
        size_t it = 0;
        weight* tbp_ptr = p_data.trained_bit_pool;
        fweight* float_tbp_ptr = p_data.float_trained_bit_pool;

        DEBUG_LOG("train state\n");
        while (it < p_data.trained_bit_pool_size) {
            tbp_ptr = p_data.trained_bit_pool + (it/float_size);
            DEBUG_LOG("it: %zu, num %hhu,%hhu,%hhu,%hhu, float %f,%f,%f,%f\n", it, tbp_ptr->r, tbp_ptr->g, tbp_ptr->b, tbp_ptr->a, float_tbp_ptr->r, float_tbp_ptr->g, float_tbp_ptr->b, float_tbp_ptr->a);
            float_tbp_ptr++;
            it += float_size;
        }

        tbp_ptr = p_data.trained_bit_pool;
        for (auto layer : p_data.layers) {

            for (size_t out_vertex_it = 0; out_vertex_it < layer->out_vertex_count; out_vertex_it++) {
                DEBUG_LOG("%1.10f\t", layer->voters_volume[out_vertex_it]);
            }
            DEBUG_LOG("\n");
        }
    }
}

void Perceptron::set_train_state_bit(LayerData& layer_data, weight* trained_bit_ptr, unsigned char value, int value_num) {
    size_t bit_num = ((size_t)trained_bit_ptr - (size_t)p_data.trained_bit_pool);
    size_t before_out_vertex_sum = 0;
    float* new_vote_sum_ptr = NULL;

    ((unsigned char*)trained_bit_ptr)[value_num] = value;

    float* float_value_ptr = &((float*)p_data.float_trained_bit_pool)[bit_num + value_num];
    *float_value_ptr = value_num == 3 ? ((unsigned char*)trained_bit_ptr)[value_num] / 255.f : char_to_float(((unsigned char*)trained_bit_ptr)[value_num]);

    if (value_num == 3) {
        size_t weight_layer_begin_diff = ((bit_num/4 - (layer_data.weights_ptr - p_data.trained_bit_pool)) % layer_data.out_vertex_count);
        // size_t weight_layers_begin_diff = weight_layer_begin_diff + (layer.weights_ptr - ptrs.h_trained_bit_pool);
        before_out_vertex_sum = (layer_data.voters_volume - p_data.voters_volume) + weight_layer_begin_diff;
        new_vote_sum_ptr = &p_data.voters_volume[before_out_vertex_sum];
        *new_vote_sum_ptr = 0.;

        fweight* t_ptr = layer_data.ptr;
        t_ptr += weight_layer_begin_diff;
        // DEBUG_LOG("weight_layer_begin_diff: %zu %p\n", weight_layer_begin_diff, t_ptr);
        for (size_t cur_vertex_it = 0; cur_vertex_it < layer_data.vertex_count; cur_vertex_it++) {
            // DEBUG_LOG("%f + %f\n", *new_vote_sum_ptr, t_ptr->a);
            *new_vote_sum_ptr += t_ptr->a;
            t_ptr += layer_data.out_vertex_count;
        }

        // DEBUG_LOG("\n\n\n\n\n\nnew_vote_sum_ptr: %f (%f + %f)\n\n\n\n\n\n", new_vote_sum_ptr, p_data.voters_volume[before_out_vertex_sum], ((value / 255.f) - (((unsigned char*)trained_bit_ptr)[value_num] / 255.f)));
        // p_data.voters_volume[before_out_vertex_sum] = p_data.voters_volume[before_out_vertex_sum] + ((value / 255.f) - (((unsigned char*)trained_bit_ptr)[value_num] / 255.f));
    }

    if (p_data.cuda_ptrs.num != -1) {

        CUDA_update_train_byte_async(
            p_data.cuda_ptrs,
            bit_num,
            float_value_ptr,
            layer_data,
            before_out_vertex_sum,
            value_num,
            new_vote_sum_ptr
        );

    }
}

void Perceptron::set_train_state_byte(LayerData& layer_data, weight* trained_bit_ptr, weight& value) {
    for (int i = 0; i < 4; i++)
        set_train_state_bit(layer_data, trained_bit_ptr, ((unsigned char*)&value)[i], i);
}

void Perceptron::run(float* input, float* output) {
    if (p_data.cuda_ptrs.num != -1) {
        // думаю запуски не параллелить, т.к. вершин могут быть тысячи, но так то можно:)
        // TODO
        CUDA_RUN(p_data.layers, p_data.cuda_ptrs, p_data.working_bit_pool_size, input, output);

        if (p_config.log_to_json && p_data.trained_bit_pool_users == 0)
            save_to_json(p_data.cuda_ptrs.h_work_state);
        return;
    }

    float* work_state = (float*)calloc(p_data.working_bit_pool_size, 1);

    float* w_ptr = work_state;
    fweight* t_ptr = p_data.float_trained_bit_pool;
    size_t layer_it = 0, out_layer_size, cur_layer_size = 0;

    for (size_t vertex_it = 0; vertex_it < p_data.input_size; vertex_it++)
        w_ptr[vertex_it] = input[vertex_it];

    for (auto layer : p_data.layers) {
        out_layer_size = layer->out_vertex_count;
        cur_layer_size = layer->vertex_count;

        // DEBUG_LOG("%slayer #%zu%s, vertexes: %zu\n", red, layer_it, normal, cur_layer_size);
        // DEBUG_LOG("%sstage 1: voting%s\n", red, normal);

        // в начале проходим по исходящим вершинам и голосуем за увеличение-уменьшение по квадратичной функции
        for (size_t cur_vertex_it = 0; cur_vertex_it < cur_layer_size; cur_vertex_it++) {
            // DEBUG_LOG("%svertex #%zu%s, has state: %f\n", blue, cur_vertex_it, normal, w_ptr[cur_vertex_it]);
            for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {

                float vote = input_square_progression_modifier(t_ptr[out_vertex_it].r, w_ptr[cur_vertex_it]);
                // DEBUG_LOG("%sedge #%zu%s (r %f, g: %f, b: %f) with a %f, vote (%f+%f) to %f\n", yellow, out_vertex_it, normal, t_ptr[out_vertex_it].r, t_ptr[out_vertex_it].g, t_ptr[out_vertex_it].b, t_ptr[out_vertex_it].a, vote, (t_ptr[out_vertex_it].g < 128 && t_ptr[out_vertex_it].g != 0) ? t_ptr[out_vertex_it].g * (0.5 - vote)/0.5 : 0.f, w_ptr[cur_layer_size + out_vertex_it]);
                if (t_ptr[out_vertex_it].g >= 0.) // это значит, что мы инвертируем до голосования
                    vote += t_ptr[out_vertex_it].g * (0.5 - vote)/0.5;
                w_ptr[cur_layer_size + out_vertex_it] += t_ptr[out_vertex_it].a * vote;
            }
            t_ptr += out_layer_size;
        }

        // считаем среднее арифметическое с коэффициентом
        w_ptr += cur_layer_size;
        for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {
            // DEBUG_LOG("delim %f/%f = %.10f\n", w_ptr[out_vertex_it], layer->voters_volume[out_vertex_it], w_ptr[out_vertex_it] / layer->voters_volume[out_vertex_it]);
            w_ptr[out_vertex_it] /= layer->voters_volume[out_vertex_it];
        }

        // инвертируем число по модулю
        // DEBUG_LOG("%sstage 2: post-XOR-inversion%s\n", red, normal);
        w_ptr -= cur_layer_size;
        t_ptr -= out_layer_size * cur_layer_size;
        for (size_t cur_vertex_it = 0; cur_vertex_it < cur_layer_size; cur_vertex_it++) {
            // DEBUG_LOG("%svertex #%zu%s, has state: %f\n", blue, cur_vertex_it, normal, w_ptr[cur_vertex_it]);
            for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {
                // DEBUG_LOG("%sedge #%zu%s, has weight.g %f\n", yellow, out_vertex_it, normal, char_to_float(t_ptr[out_vertex_it].g));

                if (t_ptr[out_vertex_it].g < 0.) {
                    w_ptr[cur_layer_size + out_vertex_it] = mod(w_ptr[cur_layer_size + out_vertex_it] + t_ptr[out_vertex_it].g);
                } else continue;

                // DEBUG_LOG("after XOR: %f\n", w_ptr[cur_layer_size + out_vertex_it]);
            }
            t_ptr += out_layer_size;
        }
        w_ptr += cur_layer_size;

        layer_it++;
        cur_layer_size = out_layer_size;
    }

    for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {
        output[out_vertex_it] = w_ptr[out_vertex_it];
    }

    if (p_config.log_to_json && p_data.trained_bit_pool_users == 0)
        save_to_json(work_state);

    free(work_state);
}

float Perceptron::accuracy_check() {

    // подразумеваем, что перед проверкой обновили тренировочные коэффициенты
    if (p_data.cuda_ptrs.num != -1)
        return CUDA_ACCURACY_CHECK(
            p_data.layers,
            p_data.cuda_ptrs,
            p_data.receptions_bit_pool,
            p_data.stream_group->d_work_state,
            p_data.working_bit_pool_size/4,
            p_data.stream_group->h_input,
            p_data.input_size,
            p_data.stream_group->h_output,
            p_data.output_size,
            p_data.stream_group->nstreams);

    float accuracy = 0.;

    float input[p_data.input_size];
    float output[p_data.output_size];

    unsigned char* bytes_ptr = (unsigned char*) p_data.receptions_bit_pool;
    for (size_t train_it_it = 0; train_it_it < p_data.train_it_count; train_it_it++) {

        for (size_t bytes_it = 0; bytes_it < p_data.input_size; bytes_it++)
            input[bytes_it] = bytes_ptr[bytes_it] / 255.;

        run(input, output);

        float run_accuracy = 0.;
        for (size_t bytes_it = 0; bytes_it < p_data.output_size; bytes_it++) {
            // DEBUG_LOG("accuracy %.15f %i ~ %.15f\n", output[bytes_it], (bytes_ptr[p_data.input_size + bytes_it] ? 1 : 0), 1.0 - mod(output[bytes_it] - (bytes_ptr[p_data.input_size + bytes_it] ? 1. : 0.)));
            if ((output[bytes_it] <= 0.5f && bytes_ptr[p_data.input_size + bytes_it] == 255)
                || (output[bytes_it] > 0.5f && bytes_ptr[p_data.input_size + bytes_it] == 0)) {
                // run_accuracy = 0.;
                // break;
                continue;
            }
            run_accuracy += 1.0 - mod(output[bytes_it] - (bytes_ptr[p_data.input_size + bytes_it] / 255.));
        }
        accuracy += run_accuracy;

        bytes_ptr += p_data.input_size + p_data.output_size;
    }

    // DEBUG_LOG("average accuracy %f/%lu = %f\n", accuracy, p_data.output_size * (p_data.train_it_count), accuracy / (p_data.output_size * (p_data.train_it_count + 1.)));
    accuracy /= p_data.output_size * (p_data.train_it_count);

    return accuracy;
}

bool Perceptron::prevent_duplicate_train_data(train_data& data) {
    unsigned char* bytes_ptr = (unsigned char*) p_data.receptions_bit_pool;
    bool not_same;
    for (size_t train_it_it = 0; train_it_it < p_data.train_it_count; train_it_it++) {
        not_same = false;

        for (size_t bytes_it = 0; bytes_it < data.first.size(); bytes_it++)
            if (bytes_ptr[bytes_it] != data.first.data()[bytes_it]) {
                not_same = true;
                break;
            }

        for (size_t bytes_it = 0; bytes_it < data.second.size(); bytes_it++)
            if (bytes_ptr[p_data.input_size + bytes_it] != data.second.data()[bytes_it]) {
                not_same = true;
                break;
            }

        bytes_ptr += p_data.input_size + p_data.output_size;
        if (not_same) continue;
        WARN("train data duplicate\n");
        return true;
    }

    return false;
}

// void Perceptron::train(std::initializer_list<train_data> list) {
//     DEBUG_LOG("train list size: %zu\n", list.size());
//     for (auto item : list) {
//         train(data);
//     }
// }

void Perceptron::prepare_train(size_t size) {
    DEBUG_LOG("\n");
    p_data.to_train = size;
    if (p_data.to_train == 0) throw "can't prepare zero trains";
    p_data.to_train--;
}

void Perceptron::train(train_data_bool bool_data) {
    train_data data{bool_data.first.size(), bool_data.second.size()};

    for (size_t bytes_it = 0; bytes_it < data.first.size(); bytes_it++)
        data.first.data()[bytes_it] = bool_data.first.data()[bytes_it] ? 255 : 0;

    for (size_t bytes_it = 0; bytes_it < data.second.size(); bytes_it++)
        data.second.data()[bytes_it] = bool_data.second.data()[bytes_it] ? 255 : 0;

    train(data);
}

void Perceptron::train(train_data& data) {
    if (data.first.size() != p_data.input_size || data.second.size() != p_data.output_size) throw "train data format not valid";

    if (prevent_duplicate_train_data(data))
        return;

    p_data.train_it_count++;

    // добавляем исходные данные тренировки в RAM (потом сохраним, чтобы не повторялись)
    p_data.receptions_bit_pool = (unsigned char*)realloc(p_data.receptions_bit_pool, (p_data.input_size + p_data.output_size) * p_data.train_it_count);
    unsigned char* bytes_ptr = (unsigned char*) p_data.receptions_bit_pool + ((p_data.input_size + p_data.output_size) * (p_data.train_it_count-1));
    for (size_t bytes_it = 0; bytes_it < data.first.size(); bytes_ptr++) {
        // printf("%hhu\n", data.first.data()[bytes_it]);
        *bytes_ptr = data.first.data()[bytes_it++];
    }

    for (size_t bytes_it = 0; bytes_it < data.second.size(); bytes_ptr++) {
        // printf("%hhu\n", data.second.data()[bytes_it]);
        *bytes_ptr = data.second.data()[bytes_it++];
    }

// TODO
if (0 != p_data.to_train--) {
    DEBUG_LOG("%smissed train %10zu\n", arrow_up, p_data.to_train);
    return;
}


    if (p_data.cuda_ptrs.num != -1) {
        p_data.stream_group = new CUDAStreamGroup;
        p_data.stream_group->nstreams = p_data.train_it_count;

        CUDA_INIT_STREAMS(
            p_data.stream_group->nstreams,
            p_data.receptions_bit_pool,
            p_data.stream_group->d_work_state,
            p_data.working_bit_pool_size/4,
            p_data.stream_group->h_input,
            p_data.input_size,
            p_data.stream_group->h_output,
            p_data.output_size);
    }

    p_data.trained_bit_pool_users++;
    switch (p_config.mode) {
        case training_mode_t::basic_brute_force:
            basic_brute_force();
        break;
        case training_mode_t::wbw_brute_force:
            wbw_brute_force();
        break;
        case training_mode_t::smart_brute_force:
            smart_brute_force_initialization();
            smart_direct_brute_force();
        break;
    }

    if (p_data.cuda_ptrs.num != -1) {
        CUDA_CLEAR_STREAMS(
            p_data.stream_group->nstreams,
            p_data.stream_group->d_work_state,
            p_data.working_bit_pool_size/4,
            p_data.stream_group->h_input,
            p_data.input_size,
            p_data.stream_group->h_output,
            p_data.output_size);
        delete p_data.stream_group;
    }

    update_run_vars();
    p_data.trained_bit_pool_users--;
}

constexpr char open_nodes[] = "{\"nodes\":[";
constexpr int open_nodes_len = sizeof(open_nodes)-1;
constexpr char open_edges[] = "],\"edges\":[";
constexpr int open_edges_len = sizeof(open_edges)-1;
constexpr char close_json[] = "]}\n";
constexpr int close_len = sizeof(close_json)-1;
void Perceptron::save_to_json(float* work_state) {

    constexpr char ext[] = ".json";
    int name_len = strlen(p_config.name);

    char* filename = new char[name_len + strlen(ext) + 5];
    std::memcpy(filename, p_config.name, name_len);

    int file_counter = 0;
    struct stat buffer;

    while (file_counter != -1) {
        sprintf(filename+name_len, "%04i\n", file_counter);
        std::memcpy(filename+name_len+4, ext, strlen(ext) + 1);

        if (stat(filename, &buffer) != 0)
            break;
        file_counter++;
    }

    int fd = open(filename
    , O_RDWR | O_CREAT
    , S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);


    ssize_t cv = write(fd, open_nodes, open_nodes_len);

    char buf[512];
    size_t layer_it = 0;

    float* work_state_ptr = (float*)work_state;
    for (auto layer_size : p_config.layer_sizes) {
        for (size_t vertex_it = 0; vertex_it < *layer_size; vertex_it++) {
            sprintf(buf, "{\"id\":\"l%zuv%zu\",\"label\":\"%f\",\"x\":%f,\"y\":-%f,\"size\":1}%s",
                layer_it, vertex_it, work_state_ptr[vertex_it], 1.0 * layer_it, 1.0 * ((vertex_it + 1.0) / (*layer_size + 1.0)) * p_data.max_layer_size, ((layer_it + 1 == p_config.layer_sizes.size()) && (vertex_it + 1) == *layer_size) ? "\0" : ",\0");
            cv += pwrite(fd, buf, strlen(buf), cv);
        }
        layer_it++;
        work_state_ptr += *layer_size;
    }

    cv += pwrite(fd, open_edges, open_edges_len, cv);

    weight* train_state_ptr = p_data.trained_bit_pool;
    size_t out_layer_size = 0, cur_layer_size = 0;
    layer_it = 0;
    for (auto layer_size : p_config.layer_sizes) {
        out_layer_size = *layer_size;
        if (cur_layer_size == 0) { // первый слой
        } else {
            for (size_t cur_vertex_it = 0; cur_vertex_it < cur_layer_size; cur_vertex_it++) {
                for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {
                    // DEBUG_LOG("%sedge #%zu%s, has weight: %f\n", yellow, out_vertex_it, normal, train_state_ptr[out_vertex_it]);
                    sprintf(buf, "{\"id\":\"e%zu-%zu-%zu\",\"source\":\"l%zuv%zu\",\"target\":\"l%zuv%zu\",\"label\":\"%0.3f, %0.3f, %0.3f, %0.3f\"}%s",
                        layer_it, cur_vertex_it, out_vertex_it,
                        layer_it - 1, cur_vertex_it, layer_it, out_vertex_it,
                        char_to_float(train_state_ptr[out_vertex_it].r),
                        char_to_float(train_state_ptr[out_vertex_it].g),
                        char_to_float(train_state_ptr[out_vertex_it].b),
                        train_state_ptr[out_vertex_it].a / 255.f,
                        (layer_it + 1 == p_config.layer_sizes.size()) && (cur_vertex_it + 1 == cur_layer_size) && (out_vertex_it + 1 == out_layer_size) ? "\0" : ",\0");
                    cv += pwrite(fd, buf, strlen(buf), cv);
                }
                train_state_ptr += out_layer_size;
            }
        }

        layer_it++;
        cur_layer_size = out_layer_size;
    }

    cv = pwrite(fd, close_json, close_len, cv);

    if (cv == -1) {
        ERROR((std::string("Can't save perceptron state to ") + p_data.filename).c_str());
    } else {
        LOG("SAVED log (%s)\n", filename);
    }

    close(fd);
}

void Perceptron::save_trained_state() {
    int config_fd = open(p_data.filename
    , O_RDWR | O_CREAT
    , S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

    ssize_t cv = write(config_fd, &p_data.train_it_count, sizeof(size_t)),
        sum = cv;

    if (cv != -1)
        cv = pwrite(config_fd, p_data.trained_bit_pool, p_data.trained_bit_pool_size, sizeof(size_t));
    sum += cv;

    if (cv != -1)
        cv = pwrite(config_fd, p_data.receptions_bit_pool, (p_data.input_size + p_data.output_size) * p_data.train_it_count, p_data.trained_bit_pool_size + sizeof(size_t));
    sum += cv;

    if (cv == -1) {
        ERROR((std::string("Can't save perceptron state to ") + p_data.filename).c_str());
    } else {
        LOG("SAVED (\"%s\", %zu bytes)\n", p_data.filename, sum);
    }
    close(config_fd);
}

bool Perceptron::is_valid_trained_bits() {
    // походу, они не могут быть теперь не валидными, посмотрим
    // for (ssize_t i = 0; i < p_data.trained_bit_pool_size; i += float_size) {
    //     // DEBUG_LOG("fnum %f ", p_data.trained_bit_pool[i/float_size]);
    //     if (p_data.trained_bit_pool[i/float_size] <= -1.0 || p_data.trained_bit_pool[i/float_size] >= 1.0) {
    //         // DEBUG_LOG("not valid\n");
    //         return false;
    //     }
    //     // DEBUG_LOG("is valid\n");
    // }
    return true;
}

bool Perceptron::is_vertex_has_positive_A(LayerData& layer, size_t vertex_num) {
    weight* layer_ptr = &layer.weights_ptr[vertex_num];
    for (size_t vertex_it = 0; vertex_it < layer.vertex_count; vertex_it++) {
        if (layer_ptr->a != 0) return true;
        layer_ptr += layer.out_vertex_count;
    }
    return false;
}

void Perceptron::update_gpu_run_state() {
    // CUDA_set_device(p_data.cuda_ptrs.num);
    CUDA_update_run_state(
        p_data.trained_bit_pool,
        p_data.trained_bit_pool_size,
        p_data.cuda_ptrs,
        p_data.out_vertex_sum,
        p_data.layers
        );
}

void Perceptron::init_gpu_device(int num) {
    DEBUG_LOG("Perceptron::init_gpu_device %i\n", num);

    if (-1 == CUDA_init_device_generic_host_mem_props(num))
        return;

    p_data.cuda_ptrs.num = num;
    p_data.cuda_ptrs.h_trained_bit_pool = p_data.trained_bit_pool;
    CUDA_init_run_state(
        p_data.trained_bit_pool,
        p_data.trained_bit_pool_size,
        p_data.cuda_ptrs,
        p_data.out_vertex_sum,
        p_data.layers
        );
    init_gpu_work_state(p_data.cuda_ptrs);
}


void Perceptron::init_gpu_work_state(cuda_gpu_device_ptrs& ptrs) {
    CUDA_alloc_work_state(
        p_data.working_bit_pool_size,
        ptrs.h_work_state
        );
}

void Perceptron::clear_gpu_device_mem() {
    // cudaDeviceReset есть еще, чтобы удалить все что создано в текущем thread
    // CUDA_set_device(p_data.cuda_ptrs.num);
    CUDA_clear_run_data(
        p_data.cuda_ptrs
        );
}
