#include <libs/neuralnet/Perceptron.h>

enum t_value {
    v,
    r,
    g,
    b,
    a
};

// считаем среднее из крайних положений коэффициентов
void Perceptron::basic_brute_force() {
    DEBUG_LOG("Perceptron::basic_brute_force\n");

    weight* best_trained_bit_pool = (weight*) calloc(p_data.trained_bit_pool_size, 1);
    weight* tb_ptr;
    char upper_symbols[p_data.trained_bytes_count * (sizeof(arrow_up) - 1)];
    for (size_t bytes_it = 0; bytes_it < p_data.trained_bytes_count + 1; bytes_it++) {
        std::memcpy(&upper_symbols[bytes_it * (sizeof(arrow_up) - 1)], arrow_up, sizeof(arrow_up) - 1);
        upper_symbols[(bytes_it+1) * (sizeof(arrow_up) - 1)] = '\0';
        DEBUG_LOG("\n");
    }

    bool not_good_accuracy = true;
    float best_accuracy = accuracy_check();

    std::memcpy(best_trained_bit_pool, p_data.trained_bit_pool, p_data.trained_bit_pool_size);

    tb_ptr = p_data.trained_bit_pool;
    for (size_t bytes_it = 0; bytes_it < p_data.trained_bytes_count; bytes_it++) {
        tb_ptr->r = 127;
        tb_ptr->g = 127;
        // tb_ptr->b = 127;
        tb_ptr->a = 255;
        // DEBUG_LOG("(r: %hhu, g: %hhu, b: %hhu, a: %hhu) ", tb_ptr->r, tb_ptr->g, tb_ptr->b, tb_ptr->a);
        tb_ptr++;
    }
    // DEBUG_LOG("\n");

    size_t train_it = 0;
    t_value current_value = t_value::g;

    while (not_good_accuracy) {
        float cur_accuracy = accuracy_check();

        if (best_accuracy < cur_accuracy) {
            best_accuracy = cur_accuracy;
            std::memcpy(best_trained_bit_pool, p_data.trained_bit_pool, p_data.trained_bit_pool_size);
        }

        fflush(stdout);
        DEBUG_LOG("%scur: %f, best: %f\n", upper_symbols, cur_accuracy, best_accuracy);

        tb_ptr = p_data.trained_bit_pool;
switch_value:
        switch (current_value) {
            case t_value::v:
                if (tb_ptr == (p_data.trained_bit_pool + p_data.trained_bytes_count - 1)) {
                    DEBUG_LOG("THE END\n");
                    not_good_accuracy = false;
                }
                tb_ptr++;
                current_value = t_value::g;
                goto switch_value;
            break;
            case t_value::r:
                if (tb_ptr->r == 255) {
                    current_value = t_value::v;
                    tb_ptr->r = 0;
                    goto switch_value;
                }
                if (tb_ptr->r < 120) {
                    tb_ptr->r += 60;
                } else if (tb_ptr->r < 135) {
                    if (tb_ptr->r == 130) {
                        tb_ptr->r = 135;
                    } else {
                        tb_ptr->r += 1;
                    }
                } else {
                    tb_ptr->r += 60;
                }
                current_value = t_value::g;
            break;
            case t_value::g:
                if (tb_ptr->g == 255) {
                    current_value = t_value::r;
                    tb_ptr->g = 0;
                    goto switch_value;
                }
                if (tb_ptr->g < 120) {
                    tb_ptr->g += 60;
                } else if (tb_ptr->g < 135) {
                    if (tb_ptr->g == 130) {
                        tb_ptr->g = 135;
                    } else {
                        tb_ptr->g += 1;
                    }
                } else {
                    tb_ptr->g += 60;
                }
                // current_value = t_value::a;
            break;
            case t_value::b:
                // if (tb_ptr->b == 255) {
                // tb_ptr->b = 0;
                // current_value = t_value::g;
                // goto switch_value;
                // }
                // tb_ptr->b += 222;
            break;
            case t_value::a:
                if (tb_ptr->a == 255) {
                    tb_ptr->a = 0;
                    current_value = t_value::b;
                    goto switch_value;
                }
                tb_ptr->a += 51;
            break;
        }

        tb_ptr = p_data.trained_bit_pool;
        for (size_t bytes_it = 0; bytes_it < p_data.trained_bytes_count; bytes_it++) {
            DEBUG_LOG("(r:%hhu g:%hhu a:%hhu)      \n", tb_ptr->r, tb_ptr->g, tb_ptr->a);
            tb_ptr++;
        }

        if (train_it++ > 200000000
            || best_accuracy == 1.0f) break;

        // if (train_it % 10000 == 0) DEBUG_LOG("\n\n\n");
    }

    for (size_t layer_it = 0; layer_it < p_config.layer_sizes.size(); layer_it++) {
        DEBUG_LOG("\n");
    }

    DEBUG_LOG("iterations count: %zu\n", train_it);

    std::memcpy(p_data.trained_bit_pool, best_trained_bit_pool, p_data.trained_bit_pool_size);

}

    // for (size_t layer_it = 0; layer_it < p_config.layer_sizes.size() - 1; layer_it++) {
    //     size_t layer_ways_per_brunch = p_data.ways.size() / (p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it+1]);
    //     tb_ptr += layer_it == 0 ? 0 : p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it-1];
    //     for (size_t from_vertex_it = 0; from_vertex_it < p_config.layer_sizes.data()[layer_it]; from_vertex_it++) {
    //         for (size_t to_vertex_it = 0; to_vertex_it < p_config.layer_sizes.data()[layer_it+1]; to_vertex_it++) {
    //             // DEBUG_LOG("l: %zu, vertex_from: %zu, vertex_to: %zu, in_bit: %f\n", layer_it, from_vertex_it, to_vertex_it, tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
    //             // tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it].r
    //             //     = (tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it].r + 1.0) * p_data.train_it_count * layer_ways_per_brunch;
    //             // DEBUG_LOG("now: %f\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
    //         }
    //     }
    // }

    // for (auto way : p_data.ways) {
    //     size_t layer_it = 0;
    //     // float coefficient = 1.0 + (data.second.data()[way->out_num] * 1.0) - (data.first.data()[way->in_num] * 1.0);
    //     // DEBUG_LOG("way: #%zu = %f ==>> #%zu = %f, c: %f\n", way->in_num, (data.first.data()[way->in_num] * 1.0), way->out_num, (data.second.data()[way->out_num] * 1.0), coefficient);
    //     for (auto edge : way->edges) {
    //         // edge->weight_ptr->r += 1;
    //         // DEBUG_LOG("weight: %f\n", *edge->weight_ptr->r);
    //         layer_it++;
    //     }
    // }
    // tb_ptr = p_data.trained_bit_pool;
    // for (size_t layer_it = 0; layer_it < p_config.layer_sizes.size() - 1; layer_it++) {
    //     size_t layer_ways_per_brunch = p_data.ways.size() / (p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it+1]);
    //     // DEBUG_LOG("layer_ways_per_brunch: %zu\n", layer_ways_per_brunch);
    //     tb_ptr += layer_it == 0 ? 0 : p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it-1];
    //     for (size_t from_vertex_it = 0; from_vertex_it < p_config.layer_sizes.data()[layer_it]; from_vertex_it++) {
    //         for (size_t to_vertex_it = 0; to_vertex_it < p_config.layer_sizes.data()[layer_it+1]; to_vertex_it++) {
    //             // DEBUG_LOG("w:%f/(train_it: %zu * w_p_b: %zu - 1), l: %zu, vertex_from: %zu, vertex_to: %zu\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it], p_data.train_it_count, layer_ways_per_brunch, layer_it, from_vertex_it, to_vertex_it);
    //             // tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it].r
    //             //     = tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it].r / (p_data.train_it_count * layer_ways_per_brunch) - 1.0;
    //             // DEBUG_LOG("out_bit: %f\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
    //         }
    //     }
    // }