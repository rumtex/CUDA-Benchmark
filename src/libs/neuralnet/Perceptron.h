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

typedef bool (*result_validate_fn)(bool* init_data, bool* result);

enum training_mode_t {
    basic_brute_force, // перебираем все возможные варианты, ищем с наименьшей ошибкой
    wbw_brute_force, // перебираем все возможные варианты, ищем с наименьшей ошибкой
};

struct PerceptronConfiguration {
    // название персептрона, должно быть уникальным в контексте вызова программы
    // можно добавлять к названию пути (/var/perceptron1 или ../perceptrons/my-perceptron)
    // расширение ".fm" добавится автоматически
    const char*             name;

    // количественная линейная конфигурация вершин графа
    array<size_t>           layer_sizes;

    // мод тренировки сети
    training_mode_t         mode = training_mode_t::basic_brute_force;

    bool                    log_to_json = false;
};

struct weight
{
    // weight(char r, char g, char b, char a) : r(r), g(g), b(b), a(a) {}
    weight(float r, float g, float b, float a) :
        r(r < 0 ? -r * 127 + 128 : r * 127),
        g(g < 0 ? -g * 127 + 128 : g * 127),
        b(b < 0 ? -b * 127 + 128 : b * 127),
        a(255*a) {}

    unsigned char   r; // коэффициент "увеличить - уменшить - оставить прежним" значение, 0 - оставить прежним, используем квадратичную функцию по 3 точкам
    unsigned char   g; // коэффициент инверсии
    unsigned char   b; // умножаем на (256 - b) / 256
    unsigned char   a; // коэффициент значимости голоса (а они могут быть не равнозначными)
};

struct edge
{
    weight*         weight_ptr;
    size_t          out_num;
    size_t          in_num;
    float           accuracy; // коэффициент соответствия результата ожиданиям для ветки
};

// weight vector
struct way
{
    array<edge>     edges;
    size_t          out_num;
    size_t          in_num;
    float           accuracy; // коэффициент соответствия результата ожиданиям для вектора (средний по всем веткам)
};

struct LayerData
 {
    size_t          vertex_count;
    size_t          out_vertex_count;

    // массив для общего объема голосов (держим в RAM, чтобы не пересчитывать при каждом запуске)
    float*          voters_volume;
 };

struct PerceptronData
{
    char*               filename;
    weight*             trained_bit_pool;
    bool*               receptions_bit_pool;
    size_t              train_it_count = 0;
    ssize_t             trained_bit_pool_users = 0; // TODO доделать thread-safe
    ssize_t             trained_bit_pool_size = 0;
    ssize_t             trained_bytes_count = 0;
    ssize_t             working_bit_pool_size = 0;
    size_t              max_layer_size = 0;
    size_t              output_size;
    size_t              input_size;
    array<way>          ways;
    array<LayerData>    layers;
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

    void init_train_vars();
    void update_train_vars();

    bool* run(array<bool> input_data);
    void run(float* input, float* output);

    float accuracy_check(train_data& data);
    bool prevent_duplicate_train_data(train_data& data);

    void basic_brute_force(train_data& data);
    void wbw_brute_force(train_data& data);

    void train(train_data& data);
    void train(std::initializer_list<train_data> list);

    void save_to_json(float* work_state);
    void save_trained_state();
    bool is_valid_trained_bits();
};
