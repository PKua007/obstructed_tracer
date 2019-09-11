/*
 * GPUGaussianMoveGenerator.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef GPUGAUSSIANMOVEGENERATOR_H_
#define GPUGAUSSIANMOVEGENERATOR_H_


#include <curand_kernel.h>

#include "random_walker/MoveGenerator.h"


class GPUGaussianMoveGenerator : public MoveGenerator {
private:
    float sigma;
    curandState *states;
    size_t numberOfTrajectories;

public:
    CUDA_DEV GPUGaussianMoveGenerator(float sigma, unsigned int seed, size_t numberOfTrajectories);
    CUDA_DEV GPUGaussianMoveGenerator(const GPUGaussianMoveGenerator &other) = delete;
    CUDA_DEV GPUGaussianMoveGenerator operator=(GPUGaussianMoveGenerator other) = delete;
    CUDA_DEV ~GPUGaussianMoveGenerator();

    CUDA_DEV Move generateMove() override;
};

#endif /* GPUGAUSSIANMOVEGENERATOR_H_ */
