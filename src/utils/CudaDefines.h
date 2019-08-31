/*
 * CudaDefines.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef CUDADEFINES_H_
#define CUDADEFINES_H_


#define CUDA_DEVICE_COMPILATION     defined(__CUDA_ARCH__)
#define CUDA_HOST_COMPILATION       !defined(__CUDA_ARCH__)
#define USING_NVCC                  defined(__CUDACC__)

#if USING_NVCC
    #define CUDA_HOSTDEV    __host__ __device__
    #define CUDA_HOST       __host__
    #define CUDA_DEV        __device__
#else // using gcc - we don't want cuda qualifiers in gcc
    #define CUDA_HOSTDEV
    #define CUDA_HOST
    #define CUDA_DEV
#endif

#define CUDA_THREAD_IDX         (blockIdx.x*blockDim.x + threadIdx.x)
#define CUDA_NUM_THREADS        (gridDim.x*blockDim.x)
#define CUDA_IS_IT_FIRST_THREAD (CUDA_THREAD_IDX == 0)

#endif /* CUDADEFINES_H_ */
