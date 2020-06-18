#include <math.h>
#include <libs/algo/modifier.h>

float input_square_progression_modifier(float coefficient, float y) {
    if (coefficient < 0)
        return -((std::pow(-coefficient, 1.3) * y) + (1-y)) + 1;
    if (coefficient == 0)
        return y;
    return (std::pow(coefficient, 1.3) * (1-y) + y);
}