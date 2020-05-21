#include <libs/algo/random.h>

int random_seed_iter = 0;
float random_unit_fraction() {
    srand(time(NULL) + (random_seed_iter * random_seed_iter));
    random_seed_iter++;
    random_seed_iter *= 19;

    unsigned randnum = random() | 2139095040; //0b0011111110000000
    randnum <<= 2;
    randnum >>= 2;
    return *reinterpret_cast<float*>(&randnum) - 1;
}

float random_sign_float_unit_fraction() {
    srand(time(NULL) + (random_seed_iter * random_seed_iter));
    random_seed_iter++;
    random_seed_iter *= 19;

    unsigned randnum = random();

    // если 1 первым байтиком
    if (randnum & 1) {
        randnum |= 2139095040;
        randnum <<= 2;
        randnum >>= 2;

        // проставляем sign 1
        randnum |= 2147483648;

        return *reinterpret_cast<float*>(&randnum) + 1;
    } else {
        randnum |= 2139095040;
        randnum <<= 2;
        randnum >>= 2;
    }

    return *reinterpret_cast<float*>(&randnum) - 1;
}