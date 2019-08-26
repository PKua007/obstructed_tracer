/*
 * CudaQualifiers.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef CUDAQUALIFIERS_H_
#define CUDAQUALIFIERS_H_

#ifdef __CUDACC__
    #define CUDA_HOSTDEV __host__ __device__
    #define CUDA_HOST __host__
    #define CUDA_DEV ___device__
#else
    #define CUDA_HOSTDEV
    #define CUDA_HOST
    #define CUDA_DEV
#endif

#endif /* CUDAQUALIFIERS_H_ */
