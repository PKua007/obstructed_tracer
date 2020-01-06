/*
 * GPUDataAccessor.cu
 *
 *  Created on: 29 gru 2019
 *      Author: pkua
 */

#include "GPUDataAccessor.h"

__global__
void gpu_copy_to_global_memory(const void *object, void *globalDest, size_t size) {
    if (!CUDA_IS_IT_FIRST_THREAD)
        return;

    memcpy(globalDest, object, size);
}
