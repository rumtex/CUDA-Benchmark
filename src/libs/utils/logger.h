#ifndef __LOGGER_H_
#define __LOGGER_H_

#include <string>

#define LOG(...) printf(__VA_ARGS__);
#define WARN(...) printf(__VA_ARGS__);

#ifdef DEBUG
#define ERROR(str) fprintf (stderr, "ERROR: %s at %s, line %d.\n", str, __FILE__, __LINE__);
#else //DEBUG
#define ERROR(str) fprintf (stderr, "ERROR: %s at %s, %s.\n", str, __DATE__, __TIME__);
#endif //DEBUG

#ifdef DEBUG
#define DEBUG_LOG(...) printf(__VA_ARGS__);
#else //DEBUG
#define DEBUG_LOG(...)
#endif //DEBUG

#endif //__LOGGER_H_