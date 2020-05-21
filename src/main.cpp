#include <libs/utils/logger.h>
#include <libs/algo/random.h>
#include <libs/neuralnet/Perceptron.h>

bool validate_result(bool* init_data, bool* result) {
    if (init_data[0]) {
        if (init_data[1]) {
            if (result[0]) {
                return true;
            } else {
                return false;
            }
        } else {
            if (result[0]) {
                return true;
            } else {
                return false;
            }
        }
    } else {
        if (init_data[1]) {
            if (result[0]) {
                return true;
            } else {
                return false;
            }
        } else {
            if (result[0]) {
                return false;
            } else {
                return true;
            }
        }
    }
    return false;
}

/*   1st    2nd      out
**          +           
**   +                  
**          +         +
**   +                  
**          +           
*/
void print_result(bool* result) {
    printf("RESULT ===>> %i\n", result[0] ? 1 : 0);
}

int main(int argc, char const *argv[])
{
    try {
        Perceptron P1 ({
                "Знаток \"ИЛИ\"",
                { 2, 3, 1 },
                .float_generator = random_sign_float_unit_fraction,
                .validator = validate_result
            });

        P1.train({
            {{true, true},{true}},
            {{true, false},{true}},
            {{false, true},{true}},
            {{false, false},{false}},
        });

        print_result(P1.run({true, true}));
        print_result(P1.run({true, false}));
        print_result(P1.run({false, true}));
        print_result(P1.run({false, false}));

    } catch (const char* err) {
        ERROR(err);
    }

    // float y = .0;
    // for (float it = -1; it < 1.01; it += 0.05) {
    //     printf("f(%f), y=%f\t = %f\n", it, y, input_square_progression_modifier(it, y));
    // }

    return 0;
}
