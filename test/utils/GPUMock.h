/*
 * GPUMock.h
 *
 *  Created on: 26 gru 2019
 *      Author: pkua
 */

#ifndef GPUMOCK_H_
#define GPUMOCK_H_

#include "utils/CudaDefines.h"


class GPUMock {
protected:

#if CUDA_DEVICE_COMPILATION
    CUDA_DEV char *getClassName0(const char *name, char *nameOut, size_t maxSize) const {
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
#else
    CUDA_DEV char *getClassName0(const char *name, char *nameOut, size_t maxSize) const {
        return nullptr;
    }
#endif

public:
    CUDA_HOSTDEV virtual ~GPUMock() { }

    CUDA_DEV virtual char *getClassName(char *nameOut, size_t maxSize) const = 0;
};


#endif /* GPUMOCK_H_ */
