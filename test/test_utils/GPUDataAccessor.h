/*
 * GPUDataAccessor.h
 *
 *  Created on: 29 gru 2019
 *      Author: pkua
 */

#ifndef GPUDATAACCESSOR_H_
#define GPUDATAACCESSOR_H_

#include <memory>
#include <vector>

#include "utils/CudaCheck.h"

__global__
void gpu_copy_object_to_global_memory(const void *object, void *globalDest, size_t size);

template<typename T>
struct CharMemoryDeleter {
    void operator()(T *ptr) const {
        char *charPtr = reinterpret_cast<char*>(ptr);
        delete [] charPtr;
    }
};

template<typename Derived, typename Base = Derived>
std::unique_ptr<Derived, CharMemoryDeleter<Derived>> get_gpu_data_accessor(const Base *gpuObject) {
    Derived *gpuGlobalMemoryObject;
    cudaCheck( cudaMalloc(&gpuGlobalMemoryObject, sizeof(Derived)) );
    gpu_copy_object_to_global_memory<<<1, 32>>>(gpuObject, gpuGlobalMemoryObject, sizeof(Derived));

    char *memory = new char[sizeof(Derived)];
    auto cpuObject = reinterpret_cast<Derived*>(memory);
    cudaCheck( cudaMemcpy(cpuObject, gpuGlobalMemoryObject, sizeof(Derived), cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(gpuGlobalMemoryObject) );

    return std::unique_ptr<Derived, CharMemoryDeleter<Derived>>(cpuObject);
}

template<typename T>
std::unique_ptr<T, CharMemoryDeleter<T>> get_gpu_data_accessor(const T *gpuObject) {
    return get_gpu_data_accessor<T, T>(gpuObject);
}

template<typename T>
std::vector<T> get_gpu_array_as_vector(const T *gpuArray, std::size_t size) {
    T *gpuGlobalMemoryArray;
    cudaCheck( cudaMalloc(&gpuGlobalMemoryArray, sizeof(T)*size) );
    gpu_copy_object_to_global_memory<<<1, 32>>>(gpuArray, gpuGlobalMemoryArray, sizeof(T)*size);

    std::vector<T> cpuVector(size);
    cudaCheck( cudaMemcpy(cpuVector.data(), gpuGlobalMemoryArray, sizeof(T)*size, cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(gpuGlobalMemoryArray) );

    return cpuVector;
}

#endif /* GPUDATAACCESSOR_H_ */
