#include <libs/neuralnet/Perceptron.h>

// считаем среднее из крайних положений коэффициентов, но добавили ассиметрию (эффект больше тем, чем входящий бит ближе к краю)
void Perceptron::arithmetical_mean_v2_train(train_data& data) {
    float* tb_ptr = p_data.trained_bit_pool;

    for (ssize_t layer_it = 0; layer_it < p_config.layer_sizes.size() - 1; layer_it++) {
        ssize_t layer_ways_per_brunch = p_data.ways.size() / (p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it+1]);
        tb_ptr += layer_it == 0 ? 0 : p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it-1];
        for (ssize_t from_vertex_it = 0; from_vertex_it < p_config.layer_sizes.data()[layer_it]; from_vertex_it++) {
            for (ssize_t to_vertex_it = 0; to_vertex_it < p_config.layer_sizes.data()[layer_it+1]; to_vertex_it++) {
                // DEBUG_LOG("l: %zu, vertex_from: %zu, vertex_to: %zu, in_bit: %f\n", layer_it, from_vertex_it, to_vertex_it, tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
                tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]
                    = (tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it] + 1.0) * p_data.train_it_count * layer_ways_per_brunch;
                // DEBUG_LOG("now: %f\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
            }
        }
    }

    for (auto way : p_data.ways) {
        ssize_t layer_it = 0;
        float coefficient = 1.0 + (data.second.data()[way->out_num] * 1.0) - (data.first.data()[way->in_num] * 1.0);
        // DEBUG_LOG("way: #%zu = %f ==>> #%zu = %f, c: %f\n", way->in_num, (data.first.data()[way->in_num] * 1.0), way->out_num, (data.second.data()[way->out_num] * 1.0), coefficient);
        for (auto edge : way->edges) {
            *edge->weight_ptr += coefficient * edge->length;
            // DEBUG_LOG("l: %zu, %zu => %zu, weight: %f, edge_coefficient:\n", layer_it, edge->in_num, edge->out_num, *edge->weight_ptr);
            layer_it++;
        }
    }

    p_data.train_it_count++;
    tb_ptr = p_data.trained_bit_pool;
    for (ssize_t layer_it = 0; layer_it < p_config.layer_sizes.size() - 1; layer_it++) {
        ssize_t layer_ways_per_brunch = p_data.ways.size() / (p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it+1]);
        // DEBUG_LOG("layer_ways_per_brunch: %zu\n", layer_ways_per_brunch);
        tb_ptr += layer_it == 0 ? 0 : p_config.layer_sizes.data()[layer_it] * p_config.layer_sizes.data()[layer_it-1];
        for (ssize_t from_vertex_it = 0; from_vertex_it < p_config.layer_sizes.data()[layer_it]; from_vertex_it++) {
            for (ssize_t to_vertex_it = 0; to_vertex_it < p_config.layer_sizes.data()[layer_it+1]; to_vertex_it++) {
                // DEBUG_LOG("w:%f/(train_it: %zu * w_p_b: %zu - 1), l: %zu, vertex_from: %zu, vertex_to: %zu\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it], p_data.train_it_count, layer_ways_per_brunch, layer_it, from_vertex_it, to_vertex_it);
                tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]
                    = tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it] / (p_data.train_it_count * layer_ways_per_brunch) - 1.0;
                // DEBUG_LOG("out_bit: %f\n", tb_ptr[p_config.layer_sizes.data()[layer_it+1] * from_vertex_it + to_vertex_it]);
            }
        }
    }
}
