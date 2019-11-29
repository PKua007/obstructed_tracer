/*
 * GPUGaussianMoveGenerator.cpp
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#include "GPUGaussianMoveGenerator.h"

CUDA_DEV GPUGaussianMoveGenerator::GPUGaussianMoveGenerator(float sigma, unsigned int seed,
                                                            size_t numberOfTrajectories)
        // Divide sigma by sqrt(2), because if we sample x and y with sigma^2, then r is sampled from 2sigma^2
        : sigma{sigma * float{M_SQRT1_2}}, numberOfTrajectories{numberOfTrajectories}
{
    this->states = new curandState[this->numberOfTrajectories];
    for (size_t i = 0; i < numberOfTrajectories; i++)
        curand_init(seed, i, 0, &(this->states[i]));
}

CUDA_DEV GPUGaussianMoveGenerator::~GPUGaussianMoveGenerator() {
    delete [] this->states;
}

CUDA_DEV Move GPUGaussianMoveGenerator::generateMove() {
    int i = CUDA_THREAD_IDX;

    return {curand_normal(&(this->states[i])) * this->sigma, curand_normal(&(this->states[i])) * this->sigma};
}
