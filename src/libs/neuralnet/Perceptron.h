#ifndef __PERCEPTRON_H_
#define __PERCEPTRON_H_

#include <libs/algo/random.h>
#include <libs/algo/func.h>
#include <libs/algo/modifier.h>
#include <libs/utils/logger.h>
#include <libs/objects/pair.h>
#include <libs/objects/array.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

constexpr char arrow_up[] = "\033[A";
constexpr size_t float_size = sizeof(float);

typedef pair<array<unsigned char>,array<unsigned char>> train_data;
typedef pair<array<bool>,array<bool>> train_data_bool;

// typedef bool (*result_validate_fn)(bool* init_data, bool* result);

enum training_mode_t {
    basic_brute_force, // перебираем все возможные варианты, ищем с наименьшей ошибкой
    wbw_brute_force, // перебираем все возможные варианты (один вес за другим), ищем с наименьшей ошибкой
    smart_brute_force, // перебираем все возможные варианты (один вес за другим), ищем с наименьшей ошибкой
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

    int                     gpu_num; // привязал к персептрону один ГПУ пока (над съедобным интерфейсом еще надо подумать)
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

struct fweight {
    fweight(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}
    float r,g,b,a;
};

struct cuda_gpu_device_ptrs
{
    int             num;
    weight*         d_train_state = 0;
    fweight*        d_float_train_state = 0;
    float*          d_voters_volume = 0;
    weight*         h_trained_bit_pool = 0;
    void*           stream;

    //эти отдельно надо
    float*          haa_work_state = 0;
    float*          h_work_state = 0;
};

struct edge
{
    weight*         weight_ptr;
    size_t          out_num;
    size_t          in_num;
    // float           accuracy; // коэффициент соответствия результата ожиданиям для ветки
};

// weight vector
struct way
{
    array<edge>             edges;
    size_t                  out_num;
    size_t                  in_num;
    // float                   accuracy; // коэффициент соответствия результата ожиданиям для вектора (средний по всем веткам)
};

struct LayerData
{
    size_t                  vertex_count;
    size_t                  out_vertex_count;

    // массив для общего объема голосов (держим в RAM, чтобы не пересчитывать при каждом запуске)
    weight*                 weights_ptr;
    fweight*                ptr;
    float*                  voters_volume;
};

struct CUDAStreamGroup
{
    size_t      nstreams;
    float*      d_work_state;
    float*      h_input;
    float*      h_output;
};

struct PerceptronData
{
    char*                   filename;
    weight*                 trained_bit_pool;
    fweight*                float_trained_bit_pool;
    unsigned char*          receptions_bit_pool;
    size_t                  train_it_count = 0;
    size_t                  trained_bit_pool_users = 0; // TODO доделать thread-safe
    size_t                  trained_bit_pool_size = 0;
    size_t                  trained_bytes_count = 0;
    size_t                  working_bit_pool_size = 0;
    size_t                  max_layer_size = 0;
    size_t                  out_vertex_sum = 0; // для voters_volume
    size_t                  output_size;
    size_t                  input_size;
    array<way>              ways;
    array<LayerData>        layers;
    cuda_gpu_device_ptrs    cuda_ptrs; // для модели один девайс - один персептрон и так сойдет
    CUDAStreamGroup*        stream_group;
    size_t                  to_train;
    float*                  voters_volume; // держим на че делить, чтобы не пересчитывать
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

    void init_gpu_device(int num);
    void init_gpu_work_state(cuda_gpu_device_ptrs& ptrs);
    void update_gpu_run_state();
    void clear_gpu_device_mem();

    void init_run_vars();
    void update_run_vars();
    void set_train_state_bit(LayerData& layer_data, weight* trained_bit_ptr, unsigned char value, int value_num);
    void set_train_state_byte(LayerData& layer_data, weight* trained_bit_ptr, weight& value);

    void run(array<bool> input_data, bool* result);
    void run(array<unsigned char> input_data, unsigned char* result);
    void run(float* input, float* output);

    float accuracy_check();
    bool prevent_duplicate_train_data(train_data& data);

    void basic_brute_force();
    void wbw_brute_force();
    void smart_brute_force_initialization();
    void smart_direct_brute_force();

    void prepare_train(size_t size);
    void train(train_data_bool data);
    void train(train_data& data);
    // std::initializer_list ломает alloc'и на CUD'е :D
    // void train(std::initializer_list<train_data> list);

    void save_to_json(float* work_state);
    void save_trained_state();

    void debug_log_train_state();
    bool is_valid_trained_bits();

    bool is_vertex_has_positive_A(LayerData& layer, size_t vertex_num);
};

#endif //__PERCEPTRON_H__
