#pragma once
#include <iomanip>
#include <iostream>

/*
 * likely/unlikely forces branch prediction to predict true/false.
 * This forcing behavior is only enabled if compiler is GCC or Clang.
 * Otherwise, they are simply identity macros.
 *
 * This is the Linux kernel way:
 * https://stackoverflow.com/questions/20916472/why-use-condition-instead-of-condition/20916491#20916491
 */
#ifndef likely
#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif
#endif

/*
 * IMPRINT_STRONG_INLINE is a stronger version of the inline,
 * using __forceinline on MSVC, always_inline on GCC/clang, and otherwise just
 * use inline.
 */
#ifndef IMPRINT_STRONG_INLINE
#if defined(_MSC_VER)
#define IMPRINT_STRONG_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define IMPRINT_STRONG_INLINE __attribute__((always_inline)) inline
#else
#define IMPRINT_STRONG_INLINE inline
#endif
#endif

#ifndef PRINT
#define PRINT(t)                                                         \
    (std::cout << std::setprecision(9) << __LINE__ << ": " << #t << '\n' \
               << t << "\n"                                              \
               << std::endl)
#endif

#ifndef ASSERT_GOOD
#define ASSERT_GOOD(t) \
    assert(!t.array().isNaN().any() && !t.array().isInf().any())
#endif
