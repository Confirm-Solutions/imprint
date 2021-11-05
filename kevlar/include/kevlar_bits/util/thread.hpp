#pragma once
#include <cstddef>
#ifdef KEVLAR_HAS_PTHREAD
#include <pthread.h>
#endif

namespace kevlar {

int set_affinity(size_t cpu_id)
{
#if defined(KEVLAR_HAS_PTHREAD)
    cpu_set_t mask;
    int status;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
#else
    return 0;
#endif
}

} // namespace kevlar
