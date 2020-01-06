/*
 * GPUNamedClass.h
 *
 *  Created on: 26 gru 2019
 *      Author: pkua
 */

#ifndef GPUNAMEDCLASS_H_
#define GPUNAMEDCLASS_H_

/** @file */

#include <string>

#include "utils/CudaDefines.h"
#include "utils/CudaCheck.h"

/**
 * @brief An interface for names gpu classes.
 *
 * One should just inherit from it and use macro:
 * \code
 * class MyClass : public GPUNamedClass {
 * public:
 *     CUDA_IMPLEMENT_GET_CLASS_NAME("MyClass");
 * }
 * \endcode
 */
class GPUNamedClass {
protected:
    CUDA_DEV char *getClassName0(const char *name, char *nameOut, size_t maxSize) const;

public:
    CUDA_HOSTDEV virtual ~GPUNamedClass() { }

    CUDA_DEV virtual char *getClassName(char *nameOut, size_t maxSize) const = 0;
};

/**
 * @brief see GPUNamedClass
 */
#define CUDA_IMPLEMENT_GET_CLASS_NAME(name) \
    CUDA_DEV char *getClassName(char *nameOut, size_t maxSize) const override { \
        return this->getClassName0(name, nameOut, maxSize); \
    }


#if CUDA_USING_NVCC
    __global__
    void gpu_get_name_of_gpu_mock(GPUNamedClass *mock, char *name, size_t maxSize);

    template<typename ConcreteGPUMock>
    std::string get_name_of_gpu_mock(void *mock) {
        constexpr std::size_t maxSize = 64;
        char *gpuName;
        cudaCheck( cudaMalloc(&gpuName, sizeof(char) * maxSize) );
        auto *mockCasted = static_cast<ConcreteGPUMock*>(mock);
        gpu_get_name_of_gpu_mock<<<1, 32>>>(mockCasted, gpuName, maxSize);
        cudaCheck( cudaDeviceSynchronize() );
        char cpuName[maxSize];
        cudaCheck( cudaMemcpy(cpuName, gpuName, sizeof(char) * maxSize, cudaMemcpyDeviceToHost) );
        cudaCheck( cudaFree(gpuName) );
        return cpuName;
    }

    #ifndef STRINGIFY
        #define XSTRINGIFY(a) STRINGIFY(a)
        #define STRINGIFY(a) #a
    #endif

    #define CUDA_IS_INSTANCE_OF(obj, clazz) (get_name_of_gpu_mock<clazz>(obj) == STRINGIFY(clazz))
#endif


#endif /* GPUNAMEDCLASS_H_ */
