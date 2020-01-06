/*
 * GPURTTI.cu
 *
 *  Created on: 29 gru 2019
 *      Author: pkua
 */

#include "GPURTTI.h"


__global__
void gpu_get_name_of_gpu_mock(GPUNamedClass *mock, char *name, size_t maxSize) {
    if (!CUDA_IS_IT_FIRST_THREAD)
        return;

    mock->getClassName(name, maxSize);
}

CUDA_DEV char *GPUNamedClass::getClassName0(const char *name, char *nameOut, size_t maxSize) const {
    size_t i = 0;
    do {
        if (i == maxSize) {
            printf("Buffer overflow when saving class name");
            asm("trap;");
        }
        nameOut[i] = name[i];
    } while (name[i++] != '\0');

    return nameOut;
}
