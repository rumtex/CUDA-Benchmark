#include <libs/neuralnet/Perceptron.h>
#include "libs/neuralnet/CUDA/kernel.h"
#include <chrono>
#include <thread>
/**
 * Пытаемся искать лучшую конфигурацию сети и как можем экономим время при этом
 * инициализируем тренировочную область, установив все веса в (0,0,0,1), которые передают свойство без изменений
 *
 * перебирая веса один за одним рискуем упустить эффект от комбинаций - исключено
 * перебирая веса попарно в пределах одного слоя так же рискуем пропустить при этом эффект от комбинаций с другим слоем - исключено
 *
 * Перебираем векторы коэффициентов (векторных :)) (далее - way) попарно в нескольких крайних положениях, затем подводим к значениям с наименьшей ошибкой
 * пропускаем при этом смежные для двух путей векторые коэффициенты, чтобы не делать лишнюю работу
 * 
 * Держим за аксиому:
 * - на вход одной вершины может быть не больше одного XOR коэффициента (g < 0) - накладывать больше одного пост-эффекта нет надобности
 * - на вход одной вершины должен быть, по крайней мере, один не прозрачный вес (a != 0) - чтобы избежать деления на 0, делать абсолютно нулевые вершины тоже надобности нет
 * держим в виду эти два условия.
 *
 * так же держим в уме, что коэффициент r имеет однонаправленную (квадратичную) зависимость, так что можно проверять в которую сторону он увеличивает точность (с оглядкой на соседний XOR - нет)
 */

void check_same_edges(bool has_same_edges[], way* way_a, way* way_b) {
    size_t edge_it = 0;
    for (auto edge : way_a->edges) {
        if (edge->weight_ptr == way_b->edges.data()[edge_it].weight_ptr) {
            has_same_edges[edge_it] = true;
        } else {
            has_same_edges[edge_it] = false;
        }
        edge_it++;
    }
}

enum t_value {
    next_way = -2,
    next_weight = -1,
    next_r = 0,
    next_g = 1,
    // next_b = 2,
    next_a = 3
};

enum t_cur_way {a_way,b_way};

enum t_step {
    BIT8 = 51,
    HD = 17,
    UHD = 1
};

void Perceptron::smart_brute_force_initialization() {
    DEBUG_LOG("Perceptron::smart_brute_force_initialization\n");

    weight* best_trained_bit_pool = (weight*) calloc(p_data.trained_bit_pool_size, 1);
    weight* tb_ptr;
    char upper_symbols[p_data.layers.size() * (sizeof(arrow_up) - 1) + 1];
    for (size_t bytes_it = 0; bytes_it < p_data.layers.size()  + 1; bytes_it++) {
        std::memcpy(&upper_symbols[bytes_it * (sizeof(arrow_up) - 1)], arrow_up, sizeof(arrow_up) - 1);
        upper_symbols[(bytes_it+1) * (sizeof(arrow_up) - 1)] = '\0';
        DEBUG_LOG("\n");
    }

    float best_accuracy = .0;//accuracy_check();
    std::memcpy(best_trained_bit_pool, p_data.trained_bit_pool, p_data.trained_bit_pool_size);

    tb_ptr = p_data.trained_bit_pool;
    for (size_t bytes_it = 0; bytes_it < p_data.trained_bytes_count; bytes_it++) {
        tb_ptr->r = 0;
        tb_ptr->g = 0;
        tb_ptr->b = 0;
        tb_ptr->a = 255;
        // DEBUG_LOG("(%3hhu,%3hhu,%3hhu) %p\t", tb_ptr->r, tb_ptr->g, tb_ptr->a, tb_ptr);
        tb_ptr++;
    }

    update_run_vars();

    float cur_accuracy = 0., best_ab_accuracy = 0., best_weight_accuracy = 0.;
    size_t train_it = 0, way_a_it = 0, way_b_it = 1, weight_a_it = 0, weight_b_it = 0;
    weight* best_way_a = (weight*) calloc(sizeof(weight), p_data.layers.size());
    weight* best_way_b = (weight*) calloc(sizeof(weight), p_data.layers.size());
    weight best_weight(0.,0.,0.,1);
    way* way_a = &p_data.ways.data()[way_a_it];
    way* way_b = &p_data.ways.data()[way_b_it];
    t_cur_way current_way = t_cur_way::b_way;
    t_value current_a_value = t_value::next_a, current_b_value = t_value::next_a;
    t_step current_step = t_step::BIT8;
    t_value* current_value;
    bool has_same_edges[p_data.layers.size()];
    check_same_edges(has_same_edges, way_a, way_b);

    bool do_first_stage = true;
    while (do_first_stage) {
        cur_accuracy = accuracy_check();

        if (best_accuracy < cur_accuracy) {
            best_accuracy = cur_accuracy;
            std::memcpy(best_trained_bit_pool, p_data.trained_bit_pool, p_data.trained_bit_pool_size);
        }

        if (best_ab_accuracy < cur_accuracy) {
            best_ab_accuracy = cur_accuracy;
            for (size_t i = 0; i < p_data.layers.size(); i++) {
                best_way_a[i] = *(way_a->edges.data()[i].weight_ptr);
                best_way_b[i] = *(way_b->edges.data()[i].weight_ptr);
            }
        }

        if (best_weight_accuracy < cur_accuracy) {
            best_weight_accuracy = cur_accuracy;
            best_weight = *(current_way == t_cur_way::a_way ? way_a->edges.data()[weight_a_it].weight_ptr : way_b->edges.data()[weight_b_it].weight_ptr);
        }

        // // лог текущего состояния
        // fflush(stdout);
        DEBUG_LOG("%s%zu) cur: %1.5f, best: %1.5f, A:(%1zu, %1zu) B:(%1zu, %1zu) current: %c\n", arrow_up /*upper_symbols*/, train_it, cur_accuracy, best_accuracy, way_a_it, weight_a_it, way_b_it, weight_b_it, current_way == t_cur_way::a_way ? 'A' : 'B');

        // DEBUG_LOG("%zu) cur: %1.5f, best: %1.5f, A:(%1zu, %1zu) B:(%1zu, %1zu) current: %c\n", train_it, cur_accuracy, best_accuracy, way_a_it, weight_a_it, way_b_it, weight_b_it, current_way == t_cur_way::a_way ? 'A' : 'B');

        // tb_ptr = p_data.trained_bit_pool;
        // for (auto layer : p_data.layers) {
        //     for (size_t bytes_it = 0; bytes_it < layer->vertex_count * layer->out_vertex_count; bytes_it++) {
        //         DEBUG_LOG("(%3hhu,%3hhu,%3hhu)%s\t",
        //             tb_ptr->r, tb_ptr->g, tb_ptr->a,
        //             (bytes_it+1) != layer->vertex_count * layer->out_vertex_count && (bytes_it+1) % layer->out_vertex_count == 0 ? " |" : "");
        //         tb_ptr++;
        //     }
        //     DEBUG_LOG("\n");
        // }

        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));



switch_value:
        current_value = current_way == t_cur_way::a_way
            ? &current_a_value
            : &current_b_value;
        tb_ptr = current_way == t_cur_way::a_way
            ? way_a->edges.data()[weight_a_it].weight_ptr
            : way_b->edges.data()[weight_b_it].weight_ptr;

        switch (*current_value) {
            case t_value::next_way:
                if (best_ab_accuracy != 0.) {
                    best_ab_accuracy = 0.;
                    // пишем бест вейс и переключаем b-way
                    for (size_t i = 0; i < p_data.layers.size(); i++) {
                        // *way_a->edges.data()[i].weight_ptr = best_way_a[i];
                        // *way_b->edges.data()[i].weight_ptr = best_way_b[i];
                        set_train_state_byte(p_data.layers.data()[i], way_a->edges.data()[i].weight_ptr, best_way_a[i]);
                        set_train_state_byte(p_data.layers.data()[i], way_b->edges.data()[i].weight_ptr, best_way_b[i]);
                    }
                }

                way_b_it++;
                current_way = t_cur_way::b_way;
                if (way_b_it == p_data.ways.size()) {
                    way_a_it++;
                    way_b_it = way_a_it + 1;
                    if (way_b_it == p_data.ways.size()) {
                        do_first_stage = false;
                        goto endwork;
                        // if (current_step == t_step::BIT8) {
                        //     way_a_it = 0;
                        //     way_b_it = 1;
                        //     current_step = t_step::HD;
                        // } else {
                        //     DEBUG_LOG("new circle");
                        //     exit(EXIT_FAILURE);
                        // }
                    }
                    way_a = &p_data.ways.data()[way_a_it];
                    way_b = &p_data.ways.data()[way_b_it];
                    check_same_edges(has_same_edges, way_a, way_b);
                    goto switch_value;
                }
                way_b = &p_data.ways.data()[way_b_it];
                check_same_edges(has_same_edges, way_a, way_b);

                current_a_value = t_value::next_a;
                current_b_value = t_value::next_a;
            break;
            case t_value::next_weight:
                if (best_weight_accuracy != 0.) {
                    // *tb_ptr = best_weight;
                    set_train_state_byte(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, best_weight);
                    best_weight_accuracy = 0.;
                }

                if (current_way == t_cur_way::a_way) {
                    weight_a_it++;
                    weight_b_it = 0;
                    if (weight_a_it == p_data.layers.size()) {
                        weight_a_it = 0;
                        current_a_value = t_value::next_way;
                        goto switch_value;
                    }
                } else {
                    weight_b_it++;
                    if (weight_b_it == p_data.layers.size()) {
                        weight_b_it = 0;
                        current_way = t_cur_way::a_way;
                        goto switch_value;
                    }
                }
                *current_value = t_value::next_a;
            break;
            case t_value::next_r:
                if (tb_ptr->r == 255) {
                    set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, 0, *current_value);
                    *current_value = t_value::next_weight;
                    goto switch_value;
                }
                set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, tb_ptr->r + current_step, *current_value);
                *current_value = t_value::next_a;
            break;
            case t_value::next_g:
                if (tb_ptr->g == 255) {
                    set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, 0, *current_value);
                    *current_value = t_value::next_r;
                    goto switch_value;
                }
                set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, tb_ptr->g + current_step, *current_value);

                if (tb_ptr->g > 127 && 0 != (current_way == t_cur_way::a_way ? way_a->edges.data()[weight_a_it].in_num : way_b->edges.data()[weight_b_it].in_num)) {
                    //идем мимо
                    set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, tb_ptr->g - current_step, *current_value);
                    *current_value = t_value::next_r;
                    goto switch_value;
                }

                *current_value = t_value::next_a;
            break;
            // case t_value::b:
            //     // if (tb_ptr->b == 255) {
            //     // tb_ptr->b = 0;
            //     // current_value = t_value::g;
            //     // goto switch_value;
            //     // }
            //     // tb_ptr->b += 222;
            // break;
            case t_value::next_a:
                if (current_way == t_cur_way::b_way && has_same_edges[weight_b_it]) {
                    // чтобы не ходить по смежным весам
                    current_b_value = t_value::next_weight;
                    goto switch_value;
                }
                if (tb_ptr->a == 0) {
                    set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, 255, *current_value);
                    *current_value = t_value::next_g;
                    goto switch_value;
                }
                set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, tb_ptr->a - current_step, *current_value);

                if (tb_ptr->a == 0 && !is_vertex_has_positive_A(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], current_way == t_cur_way::a_way ? way_a->edges.data()[weight_a_it].out_num : way_b->edges.data()[weight_b_it].out_num)) {
                    //идем мимо
                    set_train_state_bit(p_data.layers.data()[current_way == t_cur_way::a_way ? weight_a_it : weight_b_it], tb_ptr, 255, *current_value);
                    *current_value = t_value::next_g;
                    goto switch_value;
                }
            break;
        }


endwork:

        train_it++;
        if (best_accuracy == 1.0f) break;
    }

    DEBUG_LOG("stage 1 iterations count: %zu\n", train_it);

    std::memcpy(p_data.trained_bit_pool, best_trained_bit_pool, p_data.trained_bit_pool_size);
    update_run_vars();
    free(best_trained_bit_pool);
    free(best_way_a);
    free(best_way_b);
}
