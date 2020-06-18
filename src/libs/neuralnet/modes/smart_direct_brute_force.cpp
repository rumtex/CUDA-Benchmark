#include <libs/neuralnet/Perceptron.h>

#include <chrono>
#include <thread>

void make_new_lines() {
    for (size_t layer_it = 0; layer_it < 5; layer_it++) {
        DEBUG_LOG("\n");
    }
}

enum t_value {
    r = 0,
    g = 1,
    // b = 2,
    a = 3
};

enum t_cur_way {a_way,b_way};

enum t_step {
    BIT8 = 17,
    HD = 3,
    UHD = 1
};

enum t_direction {
    inc,
    dec
};

// const char* log_value_type(t_value current_value_t) {
//     switch (current_value_t) {
//         case t_value::next_way:
//             return "t_value::next_way";
//         break;
//         case t_value::next_weight:
//             return "t_value::next_weight";
//         break;
//         case t_value::next_r:
//             return "t_value::next_r";
//         break;
//         case t_value::next_g:
//             return "t_value::next_g";
//         break;
//         // case t_value::b:
//         //     return "t_value::b";
//         // break;
//         case t_value::next_a:
//             return "t_value::next_a";
//         break;
//     }
//     return "";
// }

void Perceptron::smart_direct_brute_force() {
    DEBUG_LOG("Perceptron::smart_direct_brute_force\n");

    weight* tb_ptr;
    char upper_symbols[p_data.layers.size() * sizeof(arrow_up) + 1];
    for (size_t bytes_it = 0; bytes_it < p_data.layers.size()  + 1; bytes_it++) {
        std::memcpy(&upper_symbols[bytes_it * (sizeof(arrow_up) - 1)], arrow_up, sizeof(arrow_up) - 1);
        upper_symbols[(bytes_it+1) * (sizeof(arrow_up) - 1)] = '\0';
        DEBUG_LOG("\n");
    }

    float best_accuracy = .0, last_accuracy = .0, cur_accuracy = .0;
    size_t train_it = 0, value_it = -1, layer_it = 0, layer_bits_it = 0;

    t_step current_step = t_step::BIT8;
    t_value current_value_t;
    unsigned char* current_value = (unsigned char*)p_data.trained_bit_pool;
    t_direction current_direction = t_direction::inc;
    current_value--;

    bool do_second_stage = true, next_value = true, next_value_accept = false, can_be_XOR = false;
    while (do_second_stage) {
        cur_accuracy = accuracy_check();

        if (best_accuracy <= cur_accuracy) {
            next_value_accept = true;
            best_accuracy = cur_accuracy;
        } else {
            next_value_accept = false;
        }

        // лог текущего состояния
        // fflush(stdout);
        tb_ptr = p_data.trained_bit_pool;
        DEBUG_LOG("%s%zu) cur: %1.5f, current_step: %i, layer: %zu, value_num: %zu\n", arrow_up /*upper_symbols*/, train_it, cur_accuracy, current_step, layer_it, value_it);
        // for (auto layer : p_data.layers) {
        //     for (size_t bytes_it = 0; bytes_it < layer->vertex_count * layer->out_vertex_count; bytes_it++) {
        //         DEBUG_LOG("(%3hhu,%3hhu,%3hhu)%s\t",
        //             tb_ptr->r, tb_ptr->g, tb_ptr->a,
        //             (bytes_it+1) != layer->vertex_count * layer->out_vertex_count && (bytes_it+1) % layer->out_vertex_count == 0 ? " |" : "");
        //         tb_ptr++;
        //     }
        //     DEBUG_LOG("\n");
        // }

        if (last_accuracy >= cur_accuracy) {
            current_direction = current_direction == t_direction::inc ? t_direction::dec : t_direction::inc;
            switch (current_step) {
                case t_step::BIT8:
                    current_step = t_step::HD;
                    // DEBUG_LOG("HD\n");
                    // make_new_lines();
                break;
                case t_step::HD:
                    current_step = t_step::UHD;
                    // DEBUG_LOG("UHD\n");
                    // make_new_lines();
                break;
                case t_step::UHD:
                    if (next_value) {
                        // sinus catched!
                        unsigned char best_value = 0;
                        unsigned char start_value = 0;
                        if (current_value_t == t_value::a && *current_value == 0
                            && !is_vertex_has_positive_A(p_data.layers.data()[layer_it],
                            ((value_it - layer_bits_it)/4) % p_data.layers.data()[layer_it].out_vertex_count)) start_value = 1;
                        for (unsigned char i = start_value;; i++) {
                            // *current_value = i;
                            set_train_state_bit(p_data.layers.data()[layer_it], (weight*)(current_value - current_value_t), i, current_value_t);
                            if (current_value_t == t_value::g && *current_value > 127 && (value_it - layer_bits_it) >= p_data.layers.data()[layer_it].out_vertex_count) break;
                            cur_accuracy = accuracy_check();
                            if (cur_accuracy >= last_accuracy) {
                                last_accuracy = cur_accuracy;
                                best_value = i;
                            }
                            if (i == 255) break;
                        }

                        // *current_value = best_value;
                        set_train_state_bit(p_data.layers.data()[layer_it], (weight*)(current_value - current_value_t), best_value, current_value_t);
                        next_value_accept = true;
                    }
                    next_value = true;
                break;
            }
        }
        last_accuracy = cur_accuracy;

        if (next_value && next_value_accept) {
            next_value = false;
next_value:
            current_value++;
            value_it++;

            if (((value_it - layer_bits_it) / (p_data.layers.data()[layer_it].vertex_count * p_data.layers.data()[layer_it].out_vertex_count * sizeof(weight))) == 1) {
                layer_bits_it += p_data.layers.data()[layer_it].vertex_count * p_data.layers.data()[layer_it].out_vertex_count * sizeof(weight);
                layer_it++;
            }
            if (p_data.trained_bit_pool_size == value_it) {
                do_second_stage = false;
                break;
            }
            switch (value_it % sizeof(weight)) {
                case 0:
                    current_value_t = t_value::r;
                break;
                case 1:
                    current_value_t = t_value::g;
                break;
                case 2:
                    goto next_value;
                    // current_value_t = t_value::b;
                break;
                case 3:
                    current_value_t = t_value::a;
                break;
            }
            if (current_value_t == t_value::g && (value_it - layer_bits_it) < p_data.layers.data()[layer_it].out_vertex_count) {
                can_be_XOR = true;
            } else {
                can_be_XOR = false;
            }

            current_step = t_step::BIT8;

            if (*current_value > 127) {
                current_direction = t_direction::dec;
            } else {
                current_direction = t_direction::inc;
            }

        }


crement:
        // old = *current_value;
        if (current_direction == t_direction::inc) {
            // *current_value += (unsigned char)current_step;
            set_train_state_bit(p_data.layers.data()[layer_it], (weight*)(current_value - current_value_t), *current_value + current_step, current_value_t);
        } else {
            // *current_value -= (unsigned char)current_step;
            set_train_state_bit(p_data.layers.data()[layer_it], (weight*)(current_value - current_value_t), *current_value - current_step, current_value_t);
        }

        if (current_value_t == t_value::a && *current_value == 0
             && !is_vertex_has_positive_A(p_data.layers.data()[layer_it],
                ((value_it - layer_bits_it)/4) % p_data.layers.data()[layer_it].out_vertex_count)) {
            goto crement;
        }

        if (current_value_t == t_value::g && *current_value > 127 && !can_be_XOR)
            goto crement;

        train_it++;
        // if (train_it == 10) exit(EXIT_SUCCESS);
        if (cur_accuracy == 1.0f) break;
    }

    DEBUG_LOG("FINISH!\n");
    DEBUG_LOG("iterations count: %zu\n", train_it);

}
