#include <libs/algo/func.h>

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