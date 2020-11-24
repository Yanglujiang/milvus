// Copyright (c) 2020 smarsu. All Rights Reserved.

#pragma once

#define PAD_UP(a, b) (((a) + (b) - 1) / (b) * (b))
#define PAD_DN(a, b) ((a) / (b) * (b))
#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define DIV_DN(a, b) ((a) / (b))

#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define ABS(a) (((a) > 0) ? (a) : (-(a)))

//#define INITINITE 0x7F800000
#define INITINITE 0xFF800000

#define ALIGN_UP PAD_UP
#define ALIGN_DN PAD_DN
#define INIFITE INITINITE

#ifdef NDEBUG

#define PERF_TIME_START()

#define PERF_TIME_END()

#define PERF_CACHE_MISS_START()

#define PERF_CACHE_MISS_END()

#define BLOG(x)

#define BCHECK(x)

#else

#define PERF_TIME_START(x) \
  struct timeval t_start##x; \
  struct timeval t_end##x; \
  __sync_all(); \
  gettimeofday(&t_start##x, NULL); \

#define PERF_TIME_END(x) \
do { \
  gettimeofday(&t_end##x, NULL); \
  __sync_all(); \
  if (taskId == 0) { \
    printf("[CNNL] %s %d: "#x" Hardware Time: %u us\n", __FILE__, __LINE__, \
      (uint32_t)t_end##x.tv_usec - (uint32_t)t_start##x.tv_usec); \
  } \
} while(0)

#define PERF_CACHE_MISS_START() \
 uint32_t m_start = 0; \
 uint32_t m_end = 0; \
 __asm__ volatile( \
   "mv.sreg.gpr %%perf_read, 1;\n\t" \
   "mv.gpr.sreg %[m_start], %%perf_cache_miss_num;\n\t" \
   : [m_start] "+&r" (m_start))

#define PERF_CACHE_MISS_END() \
 do { \
   __asm__ volatile( \
     "mv.sreg.gpr %%perf_read, 1;\n\t"  \
     "mv.gpr.sreg %[m_end], %%perf_cache_miss_num;\n\t" \
     : [m_end] "+&r" (m_end)); \
   printf("[CNNL] %s:%d: Kernel Cache Miss: %u\n", __FILE__, __LINE__, m_end - m_start); \
 } while (0)

#define BLOG(x) x

#define BCHECK(x) \
  if (!(x)) { \
    __bang_printf("[LOG ERROR] %s:%d: "#x"\n", __FILE__, __LINE__); \
    assert(x); \
  }

#endif

enum class DataType : int {
  kInvalid,
  kFloat32,
  kFloat16,
  kUint8,
  kInt8,
  kInt16,
  kInt32,
  kUint32,
};

enum Layout {
  kNCHW,
  kNHWC,
};

enum PadMode {
  kConstant,
  kEdge,
};

enum class CondtakeMode : int {
  kInvalid = -1,
  kEQ,
  kNEQ,
  kLT,
  kLEQ,
  kGT,
  kGEQ,
};
