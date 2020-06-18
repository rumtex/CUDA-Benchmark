#include <libs/neuralnet/Perceptron.h>

enum t_value {
    v,
    r,
    g,
    b,
    a
};

// считаем среднее из крайних положений коэффициентов
void Perceptron::wbw_brute_force() {
    DEBUG_LOG("Perceptron::wbw_brute_force\n");

    weight* best_trained_bit_pool = (weight*) calloc(p_data.trained_bit_pool_size, 1);
    weight* tb_ptr;
    char upper_symbols[p_data.trained_bytes_count * (sizeof(arrow_up) - 1) + 1];
    for (size_t bytes_it = 0; bytes_it < p_data.trained_bytes_count  + 1; bytes_it++) {
        std::memcpy(&upper_symbols[bytes_it * (sizeof(arrow_up) - 1)], arrow_up, sizeof(arrow_up) - 1);
        upper_symbols[(bytes_it+1) * (sizeof(arrow_up) - 1)] = '\0';
        DEBUG_LOG("\n");
    }

    bool not_good_accuracy = true;
    float best_accuracy = .0;//accuracy_check();
    std::memcpy(best_trained_bit_pool, p_data.trained_bit_pool, p_data.trained_bit_pool_size);

    tb_ptr = p_data.trained_bit_pool;
    for (size_t bytes_it = 0; bytes_it < p_data.trained_bytes_count; bytes_it++) {
        tb_ptr->r = 0;
        tb_ptr->g = 0;
        // if (bytes_it == 0) {
        // } else {
        //     tb_ptr->g = 0;
        // }
        tb_ptr->b = 0;
        tb_ptr->a = 255;
        // DEBUG_LOG("(r: %hhu, g: %hhu, b: %hhu, a: %hhu) ", tb_ptr->r, tb_ptr->g, tb_ptr->b, tb_ptr->a);
        tb_ptr++;
    }
    // DEBUG_LOG("\n");

    size_t train_it = 0, wtrain_it = 0;
    t_value current_value = t_value::a;
    weight best_weight(0,0,0,0);
    float best_weight_accuracy = 0., last_circle_accuracy = 0.;
    size_t weight_it = 0;
    bool circle_effect = false, hidpi = false;

    while (not_good_accuracy) {
        float cur_accuracy = accuracy_check();

        // return;
        if (last_circle_accuracy < cur_accuracy) {
            circle_effect = true;
            last_circle_accuracy = cur_accuracy;
        }

        if (best_accuracy < cur_accuracy) {
            best_accuracy = cur_accuracy;
            std::memcpy(best_trained_bit_pool, p_data.trained_bit_pool, p_data.trained_bit_pool_size);
        }

        tb_ptr = p_data.trained_bit_pool + weight_it;
        if (best_weight_accuracy < cur_accuracy) {
            best_weight_accuracy = cur_accuracy;
            best_weight.r = tb_ptr->r;
            best_weight.g = tb_ptr->g;
            best_weight.b = tb_ptr->b;
            best_weight.a = tb_ptr->a;
        }

        fflush(stdout);
        DEBUG_LOG("%scur: %f, best: %f\n", upper_symbols, cur_accuracy, best_accuracy);

switch_value:
        switch (current_value) {
            case t_value::v:
                weight_it++;
                if (weight_it == p_data.trained_bytes_count) {
                    weight_it = 0;
                    if (!circle_effect && hidpi)
                        not_good_accuracy = false;
                    if (!circle_effect)
                        hidpi = true;
                    circle_effect = false;
                }

                best_weight_accuracy = 0.;

                wtrain_it++;
                tb_ptr->r = best_weight.r;
                tb_ptr->g = best_weight.g;
                tb_ptr->b = best_weight.b;
                tb_ptr->a = best_weight.a;

                tb_ptr = p_data.trained_bit_pool + weight_it;
                tb_ptr->r = 0;
                tb_ptr->g = 0;
                tb_ptr->b = 127;
                tb_ptr->a = 255;

                current_value = t_value::a;
                // goto switch_value;
            break;
            case t_value::r:
                if (tb_ptr->r == 255) {
                    current_value = t_value::v;
                    tb_ptr->r = 0;
                    goto switch_value;
                }
                if (hidpi) {
                    tb_ptr->r += 1;
                } else if (tb_ptr->r < 120) {
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
                if (hidpi) {
                    tb_ptr->g += 1;
                } else if (tb_ptr->g < 120) {
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
                current_value = t_value::a;
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
                    current_value = t_value::g;
                    goto switch_value;
                }
                tb_ptr->a += 17;
            break;
        }

        tb_ptr = p_data.trained_bit_pool;
        for (size_t bytes_it = 0; bytes_it < p_data.trained_bytes_count; bytes_it++) {
            DEBUG_LOG("(r:%hhu g:%hhu a:%hhu)              \n", tb_ptr->r, tb_ptr->g, tb_ptr->a);
            tb_ptr++;
        }

        train_it++;
        if (wtrain_it > 2000
            || best_accuracy == 1.0f) break;

        // if (train_it % 10000 == 0) DEBUG_LOG("\n\n\n");
    }

    for (size_t layer_it = 0; layer_it < p_config.layer_sizes.size(); layer_it++) {
        DEBUG_LOG("\n");
    }

    DEBUG_LOG("iterations count: %zu\n", train_it);

    std::memcpy(p_data.trained_bit_pool, best_trained_bit_pool, p_data.trained_bit_pool_size);

}
