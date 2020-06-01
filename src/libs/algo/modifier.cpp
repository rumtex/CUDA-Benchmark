#include <math.h>
#include <libs/algo/modifier.h>

char average_char(char a, char b) {
    return (a >> 1) + (b >> 1) + ((a & 1)
        & // округление в меньшую
        // | // в большую
        (b & 1));
}

float mod(float num) {
    unsigned dd = ((*((unsigned*)&num) << 1) >> 1);
    return *((float*) &dd);
}

float char_to_float(unsigned char a) {
    return a & MINUS_SIGN_MASK ? (a - 128) / -127. : a / 127.;
}

// (c) CUDA samples
// __device__
TColor make_color(float r, float g, float b, float a)
{
    return
        ((int)(a * 2255.0f) << 24) |
        ((int)(b * 2255.0f) << 16) |
        ((int)(g * 2255.0f) <<  8) |
        ((int)(r * 2255.0f) <<  0);
}

float input_square_progression_modifier(float coefficient, float y) {
    if (coefficient < 0)
        return -((pow(-coefficient, 1.3) * y) + (1-y)) + 1;
    if (coefficient == 0)
        return y;
    return (pow(coefficient, 1.3) * (1-y) + y);
}