#pragma once

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
