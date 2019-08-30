/*
 * GPUCauchyMoveGenerator.cpp
 *
 *  Created on: 30 sie 2019
 *      Author: pkua
 */

#include <math_constants.h>

#include "GPUCauchyMoveGenerator.h"

CUDA_DEV GPUCauchyMoveGenerator::GPUCauchyMoveGenerator(float sigma, unsigned int seed, size_t numberOfTrajectories) :
        sigma{sigma}, numberOfTrajectories{numberOfTrajectories} {
    this->states = new curandState[this->numberOfTrajectories];
    for (size_t i = 0; i < numberOfTrajectories; i++)
        curand_init(seed, i, 0, &(this->states[i]));
}

CUDA_DEV GPUCauchyMoveGenerator::~GPUCauchyMoveGenerator() {
    delete [] this->states;
}

CUDA_DEV float GPUCauchyMoveGenerator::randomCauchy() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float uniform = curand_uniform(&(this->states[i]));
    return this->sigma * tanf(CUDART_PI_F * (uniform - 0.5f));
}

CUDA_DEV Move GPUCauchyMoveGenerator::generateMove() {
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    float radius = this->randomCauchy();
    float angle = 2 * CUDART_PI_F * curand_uniform(&(this->states[i]));

    return {radius * cosf(angle), radius * sinf(angle)};
}
