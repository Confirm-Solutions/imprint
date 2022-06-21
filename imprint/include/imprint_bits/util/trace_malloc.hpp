// Utility for counting calls to malloc
// Usage: include as first header
#include <stdlib.h>

#include <iostream>

static long numOfHeapAllocations = 0;
namespace std {
void *traced_malloc(size_t t) {
    ++numOfHeapAllocations;
    return malloc(t);
}
}  // namespace std
void *traced_malloc(size_t t) {
    ++numOfHeapAllocations;
    return malloc(t);
}

#ifndef malloc
#define malloc(t) traced_malloc(t)
#endif

class AllocCounter {
    const long startAllocations_;

   public:
    AllocCounter() : startAllocations_(numOfHeapAllocations) {}
    ~AllocCounter() {
        std::cout << "Numer of allocations: "
                  << numOfHeapAllocations - startAllocations_ << '\n';
    }
};
