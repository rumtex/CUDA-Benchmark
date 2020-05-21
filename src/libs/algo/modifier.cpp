#include <math.h>

char average_char(char a, char b) {
    return (a >> 1) + (b >> 1) + ((a & 1)
        & // округление в меньшую
        // | // в большую
        (b & 1));
}

float input_square_progression_modifier(float coefficient, float y) {
    if (coefficient < 0)
        return -((pow(-coefficient, 1.3) * y) + (1-y)) + 1;
    if (coefficient == 0)
        return y;
    return (pow(coefficient, 1.3) * (1-y) + y);
}