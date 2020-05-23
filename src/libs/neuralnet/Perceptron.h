#include <libs/algo/random.h>
#include <libs/algo/modifier.h>
#include <libs/utils/logger.h>
#include <libs/objects/pair.h>
#include <libs/objects/array.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

constexpr size_t float_size = sizeof(float);

typedef pair<array<bool>,array<bool>> train_data;

typedef float (*init_train_float_fn)();
typedef bool (*result_validate_fn)(bool* init_data, bool* result);
typedef float (*vote_fn)(float coefficient, float input);

// enum work_mode_t {
//     voiting_average, // ищет среднее арифметическое по результатам голосований на основе f(x)
//     multiplication_efficient, // перемножает на все коэффициенты
// };

enum training_mode_t {
    // тренировочные моды для voiting_average
    arithmetical_mean, // ищет среднее арифметическое значение для того чтобы задать нужный коэффициент КАЖДОМУ прошедшему нейрону в равной степени
    arithmetical_mean_v2, // среднее арифметическое с применением коэффициента близости к стороне входящей\исходящей вершины
    flash_mod, // создаем копию сетки. перебирая веса (или находя по градиентным функциям всяким) устанавливаем самые подходящие состояния, потом сравниваем их по очереди с оригиналом и выбираем вариант по пути наименьшей разницы. находим среднее или пересчитываем все с другой отправной точки
    av_backpropagation, // поиск всех ошибочных путей, исправление коэффициентов поштучно

    //тренировочные моды для перемножений
/*
** Вообще не представляю себе как это может работать частных случаях с нулем. Наверное надо всем +1 в начале..
** когда нужен инвертирсивный, по большей части, персепшн, нам должен помочь дополнительный единичный вход
** полные преобразования возможны не менее чем в два слоя и, думаю, только при помощи поиска ошибки
*/
    mp_backpropagation, // поиск всех ошибочных путей, исправление коэффициентов поштучно
};

struct PerceptronConfiguration {
    // название персептрона, должно быть уникальным в контексте вызова программы
    // можно добавлять к названию пути (/var/perceptron1 или ../perceptrons/my-perceptron)
    // расширение ".fm" добавится автоматически
    const char*             name;

    // количественная линейная конфигурация вершин графа
    array<size_t>           layer_sizes;

    // изначально коэффициенты могут быть 0 (оставить без изменений), случайными или в соответствии с особенностями задачи
    // коэффициенты должны быть в диапазоне [-1;1]
    init_train_float_fn     float_generator = random_sign_float_unit_fraction;

    // функция, которая считает "голос" из trained коэффициента и исходных данных, см. docs/three_point_square_progression.ods
    vote_fn                 voter = input_square_progression_modifier;

    // мод тренировки сети
    training_mode_t         mode = training_mode_t::arithmetical_mean;

    bool                    log_to_json = false;
};

// weight vector
struct way
{
    array<float*>   weight_ptr;
    size_t          out_num;
    size_t          in_num;
};

struct PerceptronData
{
    char*           filename;
    float*          trained_bit_pool;
    bool*           receptions_bit_pool;
    size_t          train_it_count = 0;
    ssize_t         trained_bit_pool_users = 0; // TODO доделать thread-safe
    ssize_t         trained_bit_pool_size = 0;
    ssize_t         working_bit_pool_size = 0;
    size_t          output_size;
    size_t          input_size;
    array<way>      ways;
};

/* Персептрон
** Структура, которая содержит float коэффициенты в диапазоне [-1;1] (trained bit)
** применяет эти коэффициенты на массив данных float [0,1], чтобы посчитать массив данных следующего слоя (working bit)
** может быть представлена в виде графа, где каждая вершина слоя связана с каждой вершиной соседних слоев
*/
class Perceptron
{
private:
    PerceptronConfiguration     p_config;
    PerceptronData              p_data;

public:
    Perceptron(PerceptronConfiguration p_config);
    ~Perceptron();

    bool* run(array<bool> input_data);

    void arithmetical_mean_train(train_data& data);
    void arithmetical_mean_v2_train(train_data& data);

    void train(train_data data);
    void train(array<train_data> data);

    void save_to_json(float* work_state);
    void save_trained_state();
    bool is_valid_trained_bits();
    bool prevent_duplicate_train_data(train_data data);
};
