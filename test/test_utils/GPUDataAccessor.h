/*
 * GPUDataAccessor.h
 *
 *  Created on: 29 gru 2019
 *      Author: pkua
 */

#ifndef GPUDATAACCESSOR_H_
#define GPUDATAACCESSOR_H_

/** @file */

#include <memory>
#include <vector>

#include "utils/CudaCheck.h"

__global__
void gpu_copy_to_global_memory(const void *object, void *globalDest, size_t size);

template<typename T>
struct CharMemoryDeleter {
    void operator()(T *ptr) const {
        char *charPtr = reinterpret_cast<char*>(ptr);
        delete [] charPtr;
    }
};

/**
 * @brief Copies object from gpu to cpu to access public fields
 *
 * @p It works with polymorphic objects, however one can only access public fields and invoking any methods is prohibited,
 * since virtual table is skrewed up. It can be useful for gpu mock classes which save the parameters from the
 * constructor. It handles both global and heap gpu objects.
 *
 * @p The function can also cast base class to derived class. The intended use is then:
 * \code
 *     auto accessor = get_gpu_object_accessor<Derived>(gpuBasePointer);
 * \endcode
 * Note, that one only specifies derived class type, the base class is inferred automagically.
 *
 * @p If one does not to cast the object, use with no template arguments:
 * \code
 *     auto accessor = get_gpu_object_accessor(gpuPointer);
 * \endcode
 * It will use the other overload of the function to do everything for you.
 *
 * @param gpuObject a pointer to gpu objects, on heap or in global memory
 * @tparam Derived derived class type to be specified if one wants to cast the object
 * @tparam Base base class type to be automatically inferred
 * @return Auto-memory-managed cpu accessor of gpu object
 */
template<typename Derived, typename Base = Derived>
std::unique_ptr<Derived, CharMemoryDeleter<Derived>> get_gpu_object_accessor(const Base *gpuObject) {
    Derived *gpuGlobalMemoryObject;
    cudaCheck( cudaMalloc(&gpuGlobalMemoryObject, sizeof(Derived)) );
    gpu_copy_to_global_memory<<<1, 32>>>(gpuObject, gpuGlobalMemoryObject, sizeof(Derived));

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

/**
 * @brief Is copies gpu array to cpu std::vector.
 * @param gpuArray array on gpu to be copied
 * @param size size of the array
 * @tparam type of the array
 * @return std::vector with gpu array data
 */
template<typename T>
std::vector<T> get_gpu_array_as_vector(const T *gpuArray, std::size_t size) {
    T *gpuGlobalMemoryArray;
    cudaCheck( cudaMalloc(&gpuGlobalMemoryArray, sizeof(T)*size) );
    gpu_copy_to_global_memory<<<1, 32>>>(gpuArray, gpuGlobalMemoryArray, sizeof(T)*size);

    std::vector<T> cpuVector(size);
    cudaCheck( cudaMemcpy(cpuVector.data(), gpuGlobalMemoryArray, sizeof(T)*size, cudaMemcpyDeviceToHost) );
    cudaCheck( cudaFree(gpuGlobalMemoryArray) );

    return cpuVector;
}

#endif /* GPUDATAACCESSOR_H_ */
