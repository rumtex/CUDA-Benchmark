#include <libs/algo/random.h>
#include <libs/objects/array.h>
#include <libs/utils/logger.h>

bool test_random_unit_fraction_dispersion() {
    unsigned count = 6666;

    array<float> fa(count);

    float av_float = 0, max_float = 0, min_float = 1;
    for (unsigned i = 0; i < count; i++) {
        fa.data()[i] = random_unit_fraction();
        av_float += fa.data()[i];
        if (min_float > fa.data()[i]) min_float = fa.data()[i];
        if (max_float < fa.data()[i]) max_float = fa.data()[i];
    }

    // for (unsigned i = 0; i < count; i++)
    //  DEBUG_LOG("#%i: %e (integer representation %i)\t|\t%e\t|\t%e\t|\t%e\t|\t%e\n", i, fa.data()[i], *reinterpret_cast<unsigned*>(&fa.data()[i]), do_work(fa.data()[i], -0.99), do_work(fa.data()[i], 0.01), do_work(fa.data()[i], 0.5), do_work(fa.data()[i], 0.99));

    av_float = av_float / count;
    LOG("average float: %e\n", av_float);
    LOG("min float: %e\n", min_float);
    LOG("max float: %e\n", max_float);

    LOG("total numbers generated %i\n", count);

    if (0.495f < av_float && av_float < 0.505f)
        return true;

    return false;
    // и да, дисперсию можно в сто раз лучше посчитать и визуализировать
}

bool test_random_sign_float_unit_fraction_dispersion() {
    unsigned count = 6666;

    array<float> fa(count);

    float av_float = 0, max_float = -1, min_float = 1;
    for (unsigned i = 0; i < count; i++) {
        fa.data()[i] = random_sign_float_unit_fraction();
        av_float += fa.data()[i];
        if (min_float > fa.data()[i]) min_float = fa.data()[i];
        if (max_float < fa.data()[i]) max_float = fa.data()[i];
    }

    // for (unsigned i = 0; i < count; i++)
    //  DEBUG_LOG("#%i: %e (integer representation %i)\n", i, fa.data()[i], *reinterpret_cast<unsigned*>(&fa.data()[i]));

    LOG("sum: %e\n", av_float);
    LOG("min float: %e\n", min_float);
    LOG("max float: %e\n", max_float);

    LOG("total numbers generated %i\n", count);

    if (-0.5f < av_float && av_float < 0.5f)
        return true;

    return false;
    // и да, дисперсию можно в сто раз лучше посчитать и визуализировать
}