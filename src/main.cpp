#include <libs/utils/logger.h>
#include <libs/algo/random.h>
#include <libs/neuralnet/Perceptron.h>
#include <chrono>
#include <thread>
#include <fstream>


unsigned int reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
         ((unsigned int)ch3 << 8) + ch4;
}

void RUN_MNIST() {
    unsigned char *train_input_array, *tia_ptr, *train_output_array, *toa_ptr;
    // float *input, *output;
    unsigned int magic_number = 0;
    unsigned int number_of_images = 0;
    size_t input_size;

    std::ifstream file("train-images.idx3-ubyte", std::ios::binary);
    printf("read file train-images.idx3-ubyte\n");
    if (file.is_open()) {
        unsigned char* image;
        unsigned int n_rows = 0;
        unsigned int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&number_of_images, sizeof(number_of_images));
        file.read((char*)&n_rows, sizeof(n_rows));
        file.read((char*)&n_cols, sizeof(n_cols));

        printf("magic_number %u\n", magic_number = reverse_int(magic_number));
        printf("number_of_images %u\n", number_of_images = reverse_int(number_of_images));
        printf("n_rows %u\n", n_rows = reverse_int(n_rows));
        printf("n_cols %u\n", n_cols = reverse_int(n_cols));

        input_size = n_rows * n_cols;

        train_input_array = (unsigned char*) calloc(number_of_images * input_size, sizeof(unsigned char));
        train_output_array = (unsigned char*) calloc(number_of_images * 10, sizeof(unsigned char));
        image = (unsigned char*) calloc(1, input_size * sizeof(char));

        tia_ptr = train_input_array;
        for (unsigned int i = 0; i < number_of_images; i++) {
            file.read((char*)image, sizeof(unsigned char) * n_rows * n_cols);

            for (unsigned int n_row = 0; n_row < n_rows; n_row++) {
                for (unsigned int n_col = 0; n_col < n_cols; n_col++) {
                    *tia_ptr = image[n_row * n_rows + n_col];
                    tia_ptr++;
                }
            }

        }

        free(image);

    } else {
        printf("ERROR\n");
        exit(EXIT_FAILURE);
    }
    file.close();

    file = std::ifstream("train-labels.idx1-ubyte", std::ios::binary);
    printf("read file train-labels.idx1-ubyte\n");
    if (file.is_open()) {
        unsigned int magic_number = 0;
        unsigned int label_number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&label_number_of_images, sizeof(label_number_of_images));

        printf("magic_number %u\n", magic_number = reverse_int(magic_number));
        printf("number_of_images %u\n", label_number_of_images = reverse_int(label_number_of_images));

        if (label_number_of_images != number_of_images) {
            printf("ERROR images vs labels count\n");
            exit(EXIT_FAILURE);
        }

        toa_ptr = train_output_array;
        for (unsigned int i = 0; i < number_of_images; i++) {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            for (unsigned char ii = 0; ii < 10; ii++) {
                *toa_ptr = (label == ii) ? 255 : 0;
                toa_ptr++;
            }
        }
    } else {
        printf("ERROR\n");
        exit(EXIT_FAILURE);
    }
    try {
        Perceptron percept_mnist({
                "percept_mnist_gpu",
                {
                    input_size,
                    10
                },
                .mode = training_mode_t::smart_brute_force,
                .log_to_json = true,
                .gpu_num = 0,
            });

        // number_of_images = 100;
        percept_mnist.prepare_train(number_of_images);

        tia_ptr = train_input_array;
        toa_ptr = train_output_array;
        for (unsigned int i = 0; i < number_of_images; i++) {
            train_data* data = new train_data {
                {input_size, (unsigned char*) tia_ptr},
                {10, (unsigned char*) toa_ptr}
            };
            // data->first.set_new((unsigned char*) tia_ptr);
            // data->second.set_new((unsigned char*) toa_ptr);

            percept_mnist.train(*data);

            delete data;
            tia_ptr += input_size;
            toa_ptr += 10;
        }

    } catch (const char* err) {
        ERROR(err);
    }

    tia_ptr = train_input_array;
}

void print_result(bool* result, size_t size) {
    printf("RESULT ===>> ");
    for (size_t i = 0; i < size; i++)
        printf("%i ", result[i] ? 1 : 0);

    printf("\n");
}

int main(int argc, char const *argv[])
{
    std::chrono::_V2::system_clock::time_point a_time;
    std::chrono::_V2::system_clock::time_point b_time = std::chrono::high_resolution_clock::now();

    RUN_MNIST();
    // try {
    //     size_t input_size = 2;
    //     size_t output_size = 1;

    //     Perceptron P1({
    //             "fullstack_double_operator_gpu",
    //             { input_size, 4,3,4, output_size },
    //             .mode = training_mode_t::smart_brute_force,
    //             .log_to_json = true,
    //             .gpu_num = 0
    //         });


    //     P1.prepare_train(4);

    //     P1.train(train_data_bool{{true, true},  {true}});
    //     P1.train(train_data_bool{{true, false}, {true}});
    //     P1.train(train_data_bool{{false, true}, {true}});
    //     P1.train(train_data_bool{{false, false},{false}});

    //     // P1.train({{true, true},  {false, true,   false, true,   false, true,   false, true,   false, true,   false, true,   false, true,   false, true}});
    //     // P1.train({{true, false}, {false, false,  true,  true,   false, false,  true,  true,   false, false,  true,  true,   false, false,  true,  true}});
    //     // P1.train({{false, true}, {false, false,  false, false,  true,  true,   true,  true,   false, false,  false, false,  true,  true,   true,  true}});
    //     // P1.train({{false, false},{false, false,  false, false,  false, false,  false, false,  true,  true,   true,  true,   true,  true,   true,  true}});

    //     bool* result = (bool*)calloc(output_size, sizeof(bool));
    //     P1.run({true, true}, result);
    //     print_result(result, output_size);
    //     P1.run({true, false}, result);
    //     print_result(result, output_size);
    //     P1.run({false, true}, result);
    //     print_result(result, output_size);
    //     P1.run({false, false}, result);
    //     print_result(result, output_size);

    // //     std::this_thread::sleep_for(std::chrono::seconds(1));

    // } catch (const char* err) {
    //     ERROR(err);
    // }

    a_time = std::chrono::high_resolution_clock::now();
    long sec = std::chrono::duration_cast<std::chrono::seconds>(a_time - b_time).count();
    printf("complete in %lu sec\n", sec);

    return 0;
}
