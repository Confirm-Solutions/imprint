#pragma once
#include <cstddef>
#ifdef KEVLAR_HAS_PTHREAD
#include <pthread.h>
#endif

namespace kevlar {

inline int set_affinity(size_t cpu_id)
{
#if defined(KEVLAR_HAS_PTHREAD)
    // This doesn't work on Mac OS. I doubt it's worth adding Mac CPU pinning
    // support any time in the next year, but if we ever do, this could be a useful
    // resource:
    // http://www.hybridkernel.com/2015/01/18/binding_threads_to_cores_osx.html
    // (If you're getting an error due to this line, add `-DKEVLAR_HAS_PTHREAD=OFF`
    // to your cmake call.)
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
