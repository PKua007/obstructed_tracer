/*
 * MoveGenerator.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef MOVEGENERATOR_H_
#define MOVEGENERATOR_H_

#include "Move.h"

/**
 * @brief This `__host__ __device__` interface generates random moves based on some distribution.
 *
 * It is intended to be used both on CPU and GPU, however the methods are `CUDA_DEV`, not `CUDA_HOSTDEV`, because the
 * classes consist almost exclusively from code sampling random numbers, which is completely different on CPU and GPU.
 * Therefore, CPU and GPU versions are implemented in separate files - one *.cpp and one *.cu - and are completely
 * independent classes. In *.cpp file `CUDA_DEV` unwraps to nothing and there is CPU version. In *.cu file it will give
 * us purely `__device__` class to be used on GPU.
 */
class MoveGenerator {
public:
    CUDA_DEV virtual ~MoveGenerator() { };

    /**
     * @brief Generate random move based on some distribution.
     * @return random move based on some distribution
     */
    CUDA_DEV virtual Move generateMove() = 0;
};

#endif /* MOVEGENERATOR_H_ */
