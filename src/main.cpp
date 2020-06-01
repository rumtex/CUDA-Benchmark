#include <libs/utils/logger.h>
#include <libs/algo/random.h>
#include <libs/neuralnet/Perceptron.h>

#include <chrono>

int main(int argc, char const *argv[])
{

    try {
        Perceptron P1({
                "fullstack_double_operator",
                { 2, 16 },
                .mode = training_mode_t::wbw_brute_force,
                .log_to_json = true,
            });

        std::chrono::_V2::system_clock::time_point a_time;
        std::chrono::_V2::system_clock::time_point b_time = std::chrono::high_resolution_clock::now();

        // P1.train({
        //     {{true, true},  {true}},
        //     {{true, false}, {true}},
        //     {{false, true}, {true}},
        //     {{false, false},{false}},
        // });

        P1.train({
            {{true, true},  {false, true,   false, true,   false, true,   false, true,   false, true,   false, true,   false, true,   false, true}},
            {{true, false}, {false, false,  true,  true,   false, false,  true,  true,   false, false,  true,  true,   false, false,  true,  true}},
            {{false, true}, {false, false,  false, false,  true,  true,   true,  true,   false, false,  false, false,  true,  true,   true,  true}},
            {{false, false},{false, false,  false, false,  false, false,  false, false,  true,  true,   true,  true,   true,  true,   true,  true}},
        });

        a_time = std::chrono::high_resolution_clock::now();
        long sec = std::chrono::duration_cast<std::chrono::seconds>(a_time - b_time).count();
        printf("complete in %lu sec\n", sec);


        bool* result;
        result = P1.run({true, true});
        // printf("RESULT ===>> %i\n", result[0] ? 1 : 0);
        result = P1.run({true, false});
        // printf("RESULT ===>> %i\n", result[0] ? 1 : 0);
        result = P1.run({false, true});
        // printf("RESULT ===>> %i\n", result[0] ? 1 : 0);
        result = P1.run({false, false});
        // printf("RESULT ===>> %i\n", result[0] ? 1 : 0);

    } catch (const char* err) {
        ERROR(err);
    }

    return 0;
}
