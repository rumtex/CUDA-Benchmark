#include <libs/neuralnet/Perceptron.h>

ssize_t calc_ways(PerceptronConfiguration& p_config) {
    ssize_t ways = 1;
    for (auto layer_size : p_config.layer_sizes) {
        ways *= *layer_size;
    }
    return ways;
}

Perceptron::Perceptron(PerceptronConfiguration p_config) :
    p_config{p_config},
    p_train_data{
        .ways = {calc_ways(p_config)}
    }
{
    // reserve mem blocks
    {
        size_t prev_layer_size = 0;

        DEBUG_LOG("initialize perceptron configuration...\n"
                "layers count: %zu\n"
                "layers size: ", p_config.layer_sizes.size());
        for (auto layer_size : p_config.layer_sizes) {
            if (*layer_size <= 0) throw "Perceptron config: count vertexes must be positive";
            p_data.working_bit_pool_size += *layer_size * float_size;
            if (prev_layer_size != 0 )
                p_data.trained_bit_pool_size += *layer_size * float_size * prev_layer_size;
            prev_layer_size = *layer_size;
            DEBUG_LOG("%zu ", *layer_size);
        }
        DEBUG_LOG("\n"
            "trained_bit_pool_size: %zu\n"
            "working_bit_pool_size: %zu\n"
            "weight vectors: %zu\n", p_data.trained_bit_pool_size, p_data.working_bit_pool_size, p_train_data.ways.size());

        p_data.input_size = **p_config.layer_sizes.begin();
        p_data.output_size = prev_layer_size;
        p_data.trained_bit_pool = (float*) valloc(p_data.trained_bit_pool_size);
    }

    // name && saved condition file
    {
        int name_len = strlen(p_config.name);
        if (name_len == 0) throw "Perceptron config: invalid name length";

        constexpr char ext[] = ".fm";
        p_data.filename = new char[name_len + strlen(ext) + 1];
        std::memcpy(p_data.filename, p_config.name, name_len);
        std::memcpy(p_data.filename+name_len, ext, strlen(ext) + 1);
        DEBUG_LOG("filename: \"%s\"\n", p_data.filename);

        int config_fd = open(p_data.filename
        , O_RDWR | O_CREAT
        , S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

        if (config_fd == -1) throw (std::string("Perceptron config: cant open file ") + p_data.filename).c_str();

        ssize_t cv = read(config_fd, &p_train_data.iterations, sizeof(size_t));
        DEBUG_LOG("Load %zu from file\n", cv);

        if (cv == sizeof(size_t)) {
            cv += pread(config_fd, p_data.trained_bit_pool, p_data.trained_bit_pool_size, cv);
            DEBUG_LOG("Load %zu from file\n", cv);

        } else {
            cv = -1;
        }
        if (cv == sizeof(size_t) + p_data.trained_bit_pool_size) {
            if (!is_valid_trained_bits()) cv = -1;
            if (cv != -1) {
                ssize_t recept_bits_count;
                if (p_config.disable_recept_duplicate_check) {
                    recept_bits_count = 0;
                    cv = 0;
                } else {
                    recept_bits_count = (p_data.input_size + p_data.output_size) * p_train_data.iterations;
                    p_train_data.receptions_bit_pool = (bool*)valloc(recept_bits_count);
                    cv = pread(config_fd, p_train_data.receptions_bit_pool, recept_bits_count+1, cv);
                    DEBUG_LOG("Load %zu from file\n", cv);
                }
                if (cv == recept_bits_count) {
                    LOG("Load trained state from file: %s\n", p_data.filename);
                    LOG("trained by %zu/%e receptions v%f\n", p_train_data.iterations, pow(p_data.input_size, 2), p_train_data.iterations / pow(p_data.input_size, 2));
                } else {
                    cv = -1;
                    free(p_train_data.receptions_bit_pool);
                }
            }

        } else {
            cv = -1;
        }
        close(config_fd);
        unlink(p_data.filename);

        if (cv == -1) {
            LOG("Initialize random trained state\n");
            p_train_data.iterations = 0;
            p_train_data.receptions_bit_pool = (bool*)valloc(0);
            ssize_t it = 0;
            float* ptr;
            while (it < p_data.trained_bit_pool_size) {
                ptr = p_data.trained_bit_pool + (it/float_size);
                *ptr = p_config.float_generator();

                DEBUG_LOG("it: %zu, num %f\n", it, *(float*)(p_data.trained_bit_pool + (it/float_size)));
                it += float_size;
            }
            if (!is_valid_trained_bits()) throw "Bad generating float function";
        }
    }

    // ways
    {
        size_t way_length = p_config.layer_sizes.size() - 1;
        size_t vertex_it[p_config.layer_sizes.size()];
        for (size_t it = 0; it < p_config.layer_sizes.size(); it++) {
            vertex_it[it] = 0;
        }

        float* layers_ptrs[way_length];
        for (size_t it = 0; it < way_length; it++) {
            layers_ptrs[it] = (it == 0 ? p_data.trained_bit_pool : layers_ptrs[it - 1] + p_config.layer_sizes.data()[it] * p_config.layer_sizes.data()[it - 1]);
        }

        for (auto way : p_train_data.ways) {
            way->weight_ptr = *(new array<float*>(way_length));
            ssize_t layer_it = 0;
            for (auto weight : way->weight_ptr) {
                *weight = (float*) layers_ptrs[layer_it] + vertex_it[layer_it]*p_config.layer_sizes.data()[layer_it+1] + vertex_it[layer_it + 1];

                DEBUG_LOG("layer: %zi (size: %zu), vertex_from: %zu, vertex_to: %zu, prepend %zu, %f\n",
                    layer_it, p_config.layer_sizes.data()[layer_it],
                    vertex_it[layer_it], vertex_it[layer_it + 1],
                    vertex_it[layer_it]*p_config.layer_sizes.data()[layer_it+1], **weight);
                layer_it++;
            }

            way->in_num = vertex_it[0];
            way->out_num = vertex_it[layer_it];
            while (layer_it >= 0) {
                DEBUG_LOG("l:%zu, %zu == %zu\n", layer_it, vertex_it[layer_it], p_config.layer_sizes.data()[layer_it] - 1);
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
}

Perceptron::~Perceptron()
{
    DEBUG_LOG("destroy perceptron %s\n", p_config.name);
    save_trained_state();
}

static const char red[]    = {0x1b, '[', '1', ';', '3', '1', 'm', 0};
static const char green[]  = {0x1b, '[', '1', ';', '3', '2', 'm', 0};
static const char yellow[] = {0x1b, '[', '1', ';', '3', '3', 'm', 0};
static const char blue[]   = {0x1b, '[', '1', ';', '3', '4', 'm', 0};
static const char normal[] = {0x1b, '[', '0', ';', '3', '9', 'm', 0};
bool* Perceptron::run(array<bool> input_data) {
    DEBUG_LOG("running..\ninput: ");
    for (auto byte : input_data) {
        DEBUG_LOG("%i ", *byte ? 1 : 0);
    }
    DEBUG_LOG("\n");

    static bool* result = (bool*)valloc(p_data.output_size); // sizeof(bool) == 1

    float* work_state = (float*)valloc(p_data.working_bit_pool_size);
    float accuracy = .0;
    float* w_ptr = work_state;
    float* t_ptr = p_data.trained_bit_pool;
    size_t layer_it = 0, out_layer_size, cur_layer_size = 0;

    for (auto layer_size : p_config.layer_sizes) {
        out_layer_size = *layer_size;
        if (cur_layer_size == 0) { // первый слой
            if (*layer_size != input_data.size()) throw "bad input bytes count";
            size_t input_size = 0;
            for (auto byte : input_data) {
                w_ptr[input_size++] = *byte ? 1.0 : 0.0;
            }
        } else {
            DEBUG_LOG("%slayer #%zu%s, vertexes: %zu\n", red, layer_it - 1, normal, cur_layer_size);
            for (size_t cur_vertex_it = 0; cur_vertex_it < cur_layer_size; cur_vertex_it++) {
                DEBUG_LOG("%svertex #%zu%s, has state: %f\n", blue, cur_vertex_it, normal, w_ptr[cur_vertex_it]);
                for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {
                    DEBUG_LOG("%sedge #%zu%s, has weight: %f\n", yellow, out_vertex_it, normal, t_ptr[out_vertex_it]);
                    w_ptr[cur_layer_size + out_vertex_it] += p_config.voter(t_ptr[out_vertex_it], w_ptr[cur_vertex_it]);
                }
                t_ptr += out_layer_size;
            }
            w_ptr += cur_layer_size;
            for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {
                // DEBUG_LOG("%f/%zu\n", w_ptr[out_vertex_it], cur_layer_size);
                w_ptr[out_vertex_it] /= cur_layer_size;
            }
        }

        layer_it++;
        cur_layer_size = out_layer_size;
    }

    for (size_t out_vertex_it = 0; out_vertex_it < out_layer_size; out_vertex_it++) {
        result[out_vertex_it] = (w_ptr[out_vertex_it] > 0.5 ? true : false);
        accuracy += result[out_vertex_it] ? w_ptr[out_vertex_it] : 1.0 - w_ptr[out_vertex_it];
        DEBUG_LOG("%zu. result bit %s%f%s, round for %i\n", out_vertex_it, green, w_ptr[out_vertex_it], normal, (w_ptr[out_vertex_it] > 0.5 ? 1 : 0));
    }

    accuracy /= p_data.output_size;
    LOG("accuracy: %f\n", accuracy);

    return result;
}

bool Perceptron::prevent_duplacate_train_data(train_data data) {
    WARN("Perceptron::prevent_duplacate_train_data НЕ НАПИСАН\n");
    return false;
}

void Perceptron::train(train_data data) {
    if (data.first.size() != p_data.input_size || data.second.size() != p_data.output_size) throw "train data format not valid";
    prevent_duplacate_train_data(data);

    float* tb_ptr = p_data.trained_bit_pool;

    for (ssize_t layer_it = 0; layer_it < p_config.layer_sizes.size() - 1; layer_it++) {
        ssize_t layer_ways_per_brunch = p_train_data.ways.size() / (p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it+1]);
        tb_ptr += layer_it == 0 ? 0 : p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it-1];
        for (ssize_t from_vertex_it = 0; from_vertex_it < p_config.layer_sizes.data()[layer_it]; from_vertex_it++) {
            for (ssize_t to_vertex_it = 0; to_vertex_it < p_config.layer_sizes.data()[layer_it+1]; to_vertex_it++) {
                DEBUG_LOG("l: %zu, vertex_from: %zu, vertex_to: %zu, in_bit: %f\n", layer_it, from_vertex_it, to_vertex_it, tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
                tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]
                    = (tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it] + 1.0) * p_train_data.iterations * layer_ways_per_brunch;
                DEBUG_LOG("now: %f\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
            }
        }
    }

    for (auto way : p_train_data.ways) {
        ssize_t layer_it = 0;
        float coefficient = 1.0 + (data.second.data()[way->out_num] * 1.0) - (data.first.data()[way->in_num] * 1.0);
        DEBUG_LOG("way: #%zu = %f ==>> #%zu = %f, c: %f\n", way->in_num, (data.first.data()[way->in_num] * 1.0), way->out_num, (data.second.data()[way->out_num] * 1.0), coefficient);
        for (auto weight : way->weight_ptr) {
            **weight += coefficient;
            DEBUG_LOG("weight: %f\n", **weight);
            layer_it++;
        }
    }

    p_train_data.iterations++;
    tb_ptr = p_data.trained_bit_pool;
    for (ssize_t layer_it = 0; layer_it < p_config.layer_sizes.size() - 1; layer_it++) {
        ssize_t layer_ways_per_brunch = p_train_data.ways.size() / (p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it+1]);
        DEBUG_LOG("layer_ways_per_brunch: %zu\n", layer_ways_per_brunch);
        tb_ptr += layer_it == 0 ? 0 : p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it-1];
        for (ssize_t from_vertex_it = 0; from_vertex_it < p_config.layer_sizes.data()[layer_it]; from_vertex_it++) {
            for (ssize_t to_vertex_it = 0; to_vertex_it < p_config.layer_sizes.data()[layer_it+1]; to_vertex_it++) {
                DEBUG_LOG("w:%f/(train_it: %zu * w_p_b: %zu - 1), l: %zu, vertex_from: %zu, vertex_to: %zu\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it], p_train_data.iterations, layer_ways_per_brunch, layer_it, from_vertex_it, to_vertex_it);
                tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]
                    = tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it] / (p_train_data.iterations * layer_ways_per_brunch) - 1.0;
                DEBUG_LOG("out_bit: %f\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
            }
        }
    }

    // это потом
    p_train_data.receptions_bit_pool = (bool*)realloc(p_train_data.receptions_bit_pool, (p_data.input_size + p_data.output_size) * p_train_data.iterations);
}

void Perceptron::train(array<train_data> list) {
    DEBUG_LOG("train list size: %zu\n", list.size());

    for (auto item : list) {
        train(*item);
    }
}

void Perceptron::save_trained_state() {
    int config_fd = open(p_data.filename
    , O_RDWR | O_CREAT
    , S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    DEBUG_LOG("p_train_data.iterations %zu\n", p_train_data.iterations);
    ssize_t cv = write(config_fd, &p_train_data.iterations, sizeof(size_t));
    if (cv != -1)
        cv += pwrite(config_fd, p_data.trained_bit_pool, p_data.trained_bit_pool_size, sizeof(size_t));
    if (cv == -1) {
        ERROR((std::string("Can't save perceptron state to ") + p_data.filename).c_str());
    } else {
        LOG("SAVED (\"%s\", %zu bytes)\n", p_data.filename, cv);
    }
    close(config_fd);
}

bool Perceptron::is_valid_trained_bits() {
    for (ssize_t i = 0; i < p_data.trained_bit_pool_size; i += float_size) {
        DEBUG_LOG("fnum %f ", p_data.trained_bit_pool[i/float_size]);
        if (p_data.trained_bit_pool[i/float_size] <= -1.0 || p_data.trained_bit_pool[i/float_size] >= 1.0) {
            DEBUG_LOG("not valid\n");
            return false;
        }
        DEBUG_LOG("is valid\n");
    }
    return true;
}
