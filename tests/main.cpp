#include <units/random_unit_fraction.h>

static const char red[]    = {0x1b, '[', '1', ';', '3', '1', 'm', 0};
static const char yellow[] = {0x1b, '[', '1', ';', '3', '3', 'm', 0};
static const char blue[]   = {0x1b, '[', '1', ';', '3', '4', 'm', 0};
static const char normal[] = {0x1b, '[', '0', ';', '3', '9', 'm', 0};

int main(int argc, char const *argv[])
{
    LOG("%s===>%s test_random_unit_fraction_dispersion%s\n", yellow, blue, normal);
    if (!test_random_unit_fraction_dispersion()) LOG("%sFAILED%s\n", red, normal);

    LOG("%s===>%s test_random_sign_float_unit_fraction_dispersion%s\n", yellow, blue, normal);
    if (!test_random_sign_float_unit_fraction_dispersion()) LOG("%sFAILED%s\n", red, normal);

    return 0;
}
