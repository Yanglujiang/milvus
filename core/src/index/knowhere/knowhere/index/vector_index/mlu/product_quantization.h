/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*************************************************************************/
#ifndef KERNELS_PRODUCT_QUANTIZATION_PRODUCT_QUANTIZATION_H_
#define KERNELS_PRODUCT_QUANTIZATION_PRODUCT_QUANTIZATION_H_

#include <stdint.h>

#include "cnrt.h"

__mlu_global__ void MLUUnion4ProductQuantization(void *query_vec,
                                                 void *code_vec,
                                                 void *lut,
                                                 void *lib_vec,
                                                 void *output,
                                                 cnrtDataType_t query_vec_dt,
                                                 cnrtDataType_t code_vec_dt,
                                                 cnrtDataType_t lib_vec_dt,
                                                 cnrtDataType_t out_dt,
                                                 int batch,
                                                 int m,
                                                 int k,
                                                 int D,
                                                 int n);

#endif  // KERNELS_PRODUCT_QUANTIZATION_PRODUCT_QUANTIZATION_H_
